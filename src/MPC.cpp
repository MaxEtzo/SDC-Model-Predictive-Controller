#include "MPC.h"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve.hpp>
#include "Eigen-3.3/Eigen/Core"

using CppAD::AD;

// Timestep length and duration
const size_t N 			= 10;
const double dt 		= 0.1;

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

// max speed 100mph
const double max_speed = 44.704;
// max acceleration 9.8m/s2
const double max_accel = 9.8;
// max jerk 9.8m/s3
const double max_jerk = 9.8;

// Cost coefficients
// The main idea is to approx. normalize the scales of cost elements
// e.g. (v - v_target)^2: 0 ... (100mph)^2 or (44.7m/s)^2, while cte: 0 ... (4m)^2
// additional weights to be applied: errors: 8; actuators: 5; differential actuator: 3;  
const cost_weights[7] = 

class FG_eval {
		// Private helper functions
		// target speed based on curvature and tangential acceleration
       		// total acceleration must not exceed max_accel to avoid slipping.	
		double target_speed(double R, double cur_accel){
			double max_centripetal_accel = pow(pow(max_accel, 2) - pow(cur_accel, 2), 0.5);		
			double target_v = pow(max_centripetal_accel * R, 0.5);
			if (target_v > max_speed)
				target_v = max_speed;
			return target_v;
		}

		// curvature of fitted polynomial at point x
		double curvature(double x){
			double epsilon = 0.001;
			double a = this->coeffs[2];
			double b = this->coeffs[1];
			double R = 10000;
			if (abs(a) > epsilon){
				R = pow(1 + pow(2*a*x + b,2),1.5)/abs(2*a);
			}
			return R;	
		}
	public:
		// Fitted polynomial coefficients
		Eigen::VectorXd coeffs;
		FG_eval(Eigen::VectorXd coeffs) { this->coeffs = coeffs; }

		typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
		void operator()(ADvector& fg, const ADvector& vars) {
			// MPC
			// `fg` a vector of the cost constraints, `vars` is a vector of variable values (state & actuators)
			// estimating cost
			fg[0] = 0;
			for (size_t t = 0; t < N; t++){
				fg[0] += cost_coeff[1] * pow(vars[cte_offset + t], 2); // range -3.5 .. + 3.5 [m], scale by approx. 1/7
				fg[0] += cost_coeff[2] * pow(vars[epsi_offset + t], 2); // range -pi .. + pi [rad], scale by approx. 1/2pi
				fg[0] += cost_coeff[0] * pow(vars[v_offset + t] - 55, 2); // range 0 .. 27.8 [m/s], scale ap 0.036
			}
			for (size_t t = 0; t < N - 1; t++){
				fg[0] += cost_coeff[4] * pow(vars[a_offset + t], 2); // range -3.5 .. + 3.5 [m], scale by approx. 1/7
				fg[0] += cost_coeff[3] * pow(vars[delta_offset + t], 2); // range -pi .. + pi [rad], scale by approx. 1/2pi
			}
			for (size_t t = 0; t < N - 2; t++){
				fg[0] += cost_coeff[6] * pow(vars[delta_offset + t + 1] - vars[delta_offset + t],2);
				fg[0] += cost_coeff[7] * pow(vars[a_offset + t + 1] - vars[a_offset + t], 2);
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
					a = vars[a_offset + t - 2] ;
				}
				// polyfit predictions for error estimation
				AD<double> f0 = this->coeffs[0] + this->coeffs[1] * x0 + this->coeffs[2] * pow(x0,2);
				AD<double> psi_des0 = atan(this->coeffs[1] + 2*this->coeffs[2]*x0);
				// Constraints
				fg[1 + t] = x1 - x0 - v0*cos(psi0)*dt;
				fg[1 + y_offset + t] = y1 - y0 - v0*sin(psi0)*dt;
				fg[1 + psi_offset + t] = psi1 - psi0 - v0*delta/Lf*dt;
				fg[1 + v_offset + t] = v1 - v0 - a*dt;
				fg[1 + cte_offset + t] = cte1 - f0 + y0 - v0*sin(epsi0)*dt;
				fg[1 + epsi_offset + t] = epsi1 - psi0 + psi_des0 - v0 * delta / Lf * dt;
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
		vars_lowerbound[i] = -1E20;
		vars_upperbound[i] = 1E20;	
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
