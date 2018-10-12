#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Timestep length and duration
const size_t N 			= 10;
const double dt 		= 0.10;

const size_t state_no 		= 6; // state elements number 
const size_t act_no 		= 2; // actuator elements number

// Offset indices
const size_t y_offset 		= N;
const size_t psi_offset		= 2*N;
const size_t v_offset 		= 3*N;
const size_t cte_offset		= 4*N;
const size_t epsi_offset	= 5*N;
size_t delta_offset 		= 6*N;
size_t a_offset 		= 7*N - 1;

// This value assumes the model presented in the classroom is used.
//
// It was obtained by measuring the radius formed by running the vehicle in the
// simulator around in a circle with a constant steering angle and velocity on a
// flat terrain.
//
// Lf was tuned until the the radius formed by the simulating the model
// presented in the classroom matched the previous radius.
//
// This is the length from front to CoG that has a similar radius.
const double Lf = 2.67;

// max speed aprox. 100mph
const double max_speed = 44.7;
// max acceleration 9.8m/s2
const double max_accel = 9.8; // we assume that throttle=1 corresponds to this max_accel

// Cost coefficients
// Normalize the scales of cost elements:
const double cost_weights[6] = {1, 150, 0.004, 0.001, 0.005, 0.005};

class FG_eval {
	public:
		// Fitted polynomial coefficients
		Eigen::VectorXd coeffs;
		FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

		typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
		void operator()(ADvector& fg, const ADvector& vars) {
			// MPC
			// `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
			fg[0] = 0;
			// estimating cost
			for (size_t t = 0; t < N; t++){
				fg[0] += cost_weights[0] * pow(vars[cte_offset + t], 2); 
				fg[0] += cost_weights[1] * pow(vars[epsi_offset + t], 2);
				fg[0] += cost_weights[2] * pow(vars[v_offset + t] - max_speed, 2);
			}
			// acceleration term (based on actuators)
			for (size_t t = 0; t < N - 1; t++){
				fg[0] += cost_weights[3] * pow(vars[a_offset + t]*max_accel,2);
				fg[0] += cost_weights[4] * pow(pow(vars[v_offset + t], 2) * vars[delta_offset + t] / Lf, 2); // centripetal 
			}
			// jerk term
			for (size_t t = 0; t < N - 2; t++){
				AD<double> tang_jerk = (vars[a_offset + t + 1] - vars[a_offset + t]) * max_accel / dt;
				AD<double> cp_jerk = (vars[delta_offset + t + 1] - vars[delta_offset + t]) * pow(vars[v_offset + t],2) / Lf / dt;
				fg[0] += cost_weights[5] * (pow(tang_jerk, 2) + pow(cp_jerk, 2));
			}
			fg[1] = vars[0];
			fg[1 + y_offset] = vars[y_offset];
			fg[1 + psi_offset] = vars[psi_offset];
			fg[1 + v_offset] = vars[v_offset];
			fg[1 + cte_offset] = vars[cte_offset];
			fg[1 + epsi_offset] = vars[epsi_offset];
			for (size_t t = 1; t < N; t++){
				// state at time t-1
				AD<double> x0 = vars[t - 1];
				AD<double> y0 = vars[y_offset + t - 1];
				AD<double> psi0 = vars[psi_offset + t - 1];
				AD<double> v0 = vars[v_offset + t - 1];
				AD<double> cte0 = vars[cte_offset + t - 1];
				AD<double> epsi0 = vars[epsi_offset + t - 1];
				// state at time t
				AD<double> x1 = vars[t];
				AD<double> y1 = vars[y_offset + t];
				AD<double> psi1 = vars[psi_offset + t];
				AD<double> v1 = vars[v_offset + t];
				AD<double> cte1 = vars[cte_offset + t];
				AD<double> epsi1 = vars[epsi_offset + t];
				// actuators state at time t-1
				AD<double> delta = vars[delta_offset + t - 1];
				AD<double> a = vars[a_offset + t - 1];
				// emulate delay in control 
				if (t > 1) {
					delta = vars[delta_offset + t - 2];
					// remember, translate throttle to acceleration
					a = vars[a_offset + t - 2] * max_accel;
				}
				// polyfit predictions for error estimation
				AD<double> f0 = this->coeffs[0] + this->coeffs[1] * x0 + this->coeffs[2] * pow(x0,2);
				AD<double> psi_des0 = atan(this->coeffs[1] + 2*this->coeffs[2]*x0);
				// Constraints
				fg[1 + t] = x1 - x0 - v0*cos(psi0)*dt;
				fg[1 + y_offset + t] = y1 - y0 - v0*sin(psi0)*dt;
				fg[1 + psi_offset + t] = psi1 - psi0 - v0*delta/Lf*dt;
				fg[1 + v_offset + t] = v1 - v0 - a*dt;
				fg[1 + cte_offset + t] = cte1 + f0 - y0 - v0*sin(epsi0)*dt;
				fg[1 + epsi_offset + t] = epsi1 + psi_des0 - psi0 - v0 * delta / Lf * dt;
			}
		}
};

//
// MPC class definition implementation.
//
MPC::MPC() {}
MPC::~MPC() {}

vector<double> MPC::Solve(Eigen::VectorXd state, Eigen::VectorXd coeffs) {
	bool ok = true;
	size_t i;
	typedef CPPAD_TESTVECTOR(double) Dvector;

	// Number of model variables (includes both states and inputs).
	size_t n_vars = (state_no + act_no) * N - act_no; 
	// The number of constraints
	size_t n_constraints = state_no * N;

	// Initial value of the independent variables.
	// 0 besides initial state.
	Dvector vars(n_vars);
	for (i = 0; i < n_vars; i++) {
		vars[i] = 0;
	}
	for (i = 0; i < state_no; i++){
		vars[i*N] = state[i];	
	}

	// Lower and upper limits for the variables
	Dvector vars_lowerbound(n_vars);
	Dvector vars_upperbound(n_vars);
	for (i = 0; i < n_vars; i++){
		vars_lowerbound[i] = -1E19;
		vars_upperbound[i] = 1E19;	
	}
	for (i = delta_offset; i < a_offset; i++){
		vars_lowerbound[i] = -0.436332;
		vars_upperbound[i] = 0.436332;	
	}
	for (i = a_offset; i < n_vars; i++){
		vars_lowerbound[i] = -1.0;
		vars_upperbound[i] = 1.0;
	}

	// Lower and upper limits for the constraints
	// 0 besides initial state.
	Dvector constraints_lowerbound(n_constraints);
	Dvector constraints_upperbound(n_constraints);
	for (i = 0; i < n_constraints; i++) {
		constraints_lowerbound[i] = 0;
		constraints_upperbound[i] = 0;
	}
	for (i = 0; i < state_no; i++){
		constraints_lowerbound[i*N] = state[i];
		constraints_upperbound[i*N] = state[i];
	}

	// object that computes objective and constraints
	FG_eval fg_eval(coeffs);

	// options for IPOPT solver
	std::string options;
	// Uncomment this if you'd like more print information
	options += "Integer print_level  0\n";
	// NOTE: Setting sparse to true allows the solver to take advantage
	// of sparse routines, this makes the computation MUCH FASTER. If you
	// can uncomment 1 of these and see if it makes a difference or not but
	// if you uncomment both the computation time should go up in orders of
	// magnitude.
	options += "Sparse  true        forward\n";
	options += "Sparse  true        reverse\n";
	// NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
	// Change this as you see fit.
	options += "Numeric max_cpu_time          0.5\n";

	// place to return solution
	CppAD::ipopt::solve_result<Dvector> solution;

	// solve the problem
	CppAD::ipopt::solve<Dvector, FG_eval>(
			options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
			constraints_upperbound, fg_eval, solution);
	// Check some of the solution values
	ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

	// Cost
	auto cost = solution.obj_value;
	std::cout << "Cost " << cost << std::endl;

	vector<double> res = {solution.x[delta_offset], solution.x[a_offset]};
	for (i = 0; i < N-1; i++){
		res.push_back(solution.x[i + 1]);
		res.push_back(solution.x[y_offset + i + 1]);
	}
	return res;
}
