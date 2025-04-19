% dx/dt = f(x,u)
% x is the state vector and u is the control input.

clear;clc;

% simulation setup
dt_sim = 0.001;
dt_controller = 0.05;
control_interval = floor(dt_controller/dt_sim);
simulation_step = 15000;

% MPC setup
N = 5; % prediction horizon
% dt = dt_controller; % time step
T = 10;
x0 = [1; 1]; % initial state
x_ref = [0; 0]; % reference state
u_mpc = 0; % initial control input
u_lifted = 0;
u_lifted_chopped = 0;

% Define initial guess for control input
% u_opt = zeros(N, 1);
u_opt_mpc = zeros(N, 1);
u_opt_lifted = zeros(N, 1);
u_opt_lifted_chopped = zeros(N*T, 1);
% Define lower and upper bounds for control input
lb = -0.75*ones(N,1); % lower bound for control input
ub = ones(N,1);  % upper bound for control input
lb_chopped = -0.75*ones(N*T,1); % lower bound for control input
ub_chopped = ones(N*T,1);  % upper bound for control input
% Set optimization options
options = optimoptions('fmincon', 'Display', 'off');

% Initialize arrays to store results
x_trajectory_mpc = [];
x_trajectory_lifted_mpc = [];

u_trajectory_mpc = [];
u_trajectory_lifted_mpc = [];

solve_time_mpc = zeros(simulation_step, 1);
solve_time_lifted = zeros(simulation_step, 1);

prev_solve_mpc = NaN;
prev_solve_lifted = NaN;

time_vector = (0:simulation_step - 1) * dt_sim; % Create a time vector

% Initialize two separate states for each controller
x0_mpc = x0;
x0_lifted = x0;

% state params
x1_lb = -0.1;
x1_ub = 1.3;
x2_lb = -0.4;

for t = 1:simulation_step
    if mod(t, control_interval) == 0
        % --- Standard MPC ---
        tic;
        try
            u_opt_mpc = fmincon(@(u) cost_function(u, x0_mpc, x_ref, dt_controller),...
                        u_opt_mpc, [], [], [], [], lb, ub,... 
                        @(u) nonlinear_constraints(u, x0_mpc, dt_controller, x1_lb, x1_ub, x2_lb), options);
            solve_time = toc * 1000; % in milliseconds
        catch
            solve_time = prev_solve_mpc; % fallback if fmincon fails
        end
        prev_solve_mpc = solve_time;
        solve_time_mpc(t) = solve_time;
        u_mpc = u_opt_mpc(1);

        % --- Lifted MPC ---
        tic;
        try
            u_opt_lifted = fmincon(@(u) lifted_cost_function(u, x0_lifted, x_ref, dt_controller), ...
                           u_opt_lifted, [], [], [], [], lb, ub,...
                           @(u) nonlinear_constraints(u, x0_lifted, dt_controller, x1_lb, x1_ub, x2_lb), options);
            solve_time = toc * 1000;
        catch
            solve_time = prev_solve_lifted;
        end
        prev_solve_lifted = solve_time;
        solve_time_lifted(t) = solve_time;

        u_lifted = u_opt_lifted(1);
    else
        % Hold previous solve times
        solve_time_mpc(t) = prev_solve_mpc;
        solve_time_lifted(t) = prev_solve_lifted;
    end

    % --- RK4 Integration ---
    x_next_mpc = rk4(@(x, u) nonlinear_dynamics(x, u), x0_mpc, u_mpc, dt_sim);
    x_next_lifted = rk4(@(x, u) nonlinear_dynamics(x, u), x0_lifted, u_lifted, dt_sim);

    % --- Update States and Logs ---
    x0_mpc = x_next_mpc;
    x0_lifted = x_next_lifted;

    x_trajectory_mpc = [x_trajectory_mpc; x0_mpc'];
    u_trajectory_mpc = [u_trajectory_mpc; u_mpc];
    
    x_trajectory_lifted_mpc = [x_trajectory_lifted_mpc; x0_lifted'];
    u_trajectory_lifted_mpc = [u_trajectory_lifted_mpc; u_lifted];
end

figure;
hold on;
plot(x_trajectory_mpc(:,1), x_trajectory_mpc(:,2), 'b-', 'LineWidth', 2);
plot(x_trajectory_lifted_mpc(:,1), x_trajectory_lifted_mpc(:,2), 'r--', 'LineWidth', 2);
xlabel('x_1');
ylabel('x_2');
% Add constraint bounds
xline(x1_lb, 'k-.', 'LineWidth', 1.5); % x1 lower bound
xline(x1_ub,  'k-.', 'LineWidth', 1.5); % x1 upper bound
yline(x2_lb, 'k-.', 'LineWidth', 1.5); % x2 lower bound
legend('NMPC', 'Lifted NMPC', 'Location', 'northwest');
grid on;
% matlab2tikz('vdp_1.tex');

% Second Plot: x(1) vs time
figure;
hold on;
plot(time_vector, x_trajectory_mpc(:,1), 'b-', 'LineWidth', 2);
plot(time_vector, x_trajectory_lifted_mpc(:,1), 'r--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('x_1');
% Add constraint bounds for x1
yline(x1_lb, 'k-.', 'LineWidth', 1.5);
yline(x1_ub,  'k-.', 'LineWidth', 1.5);
legend('NMPC', 'Lifted NMPC');
grid on;
% matlab2tikz('vdp_2.tex');

% Third Plot: x(2) vs time
figure;
hold on;
plot(time_vector, x_trajectory_mpc(:,2), 'b-', 'LineWidth', 2);
plot(time_vector, x_trajectory_lifted_mpc(:,2), 'r--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('x_2');
% Add constraint bounds for x2
yline(x2_lb, 'k-.', 'LineWidth', 1.5);
legend('NMPC', 'Lifted NMPC');
grid on;
% Convert the plot to TikZ
% matlab2tikz('vdp_3.tex');

% Fourth Plot: u vs time
figure;
hold on;
plot(time_vector, u_trajectory_mpc(:), 'b-', 'LineWidth', 2);
plot(time_vector, u_trajectory_lifted_mpc(:), 'r--', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Control Input: \it{u}');
legend('NMPC', 'Lifted NMPC');
grid on;
% Convert the plot to TikZ
% matlab2tikz('vdp_4.tex');

figure;
plot(time_vector, solve_time_mpc, 'b-', 'LineWidth', 2);
hold on;
plot(time_vector, solve_time_lifted, 'r-','LineWidth', 2);
xlabel('Time (s)');
ylabel('Solve Time (ms)');
legend('NMPC', 'Lifted NMPC');
grid on;
matlab2tikz('vdp_solve_time.tex');

% Runge-Kutta 4th order numerical integration method
function x_next = rk4(ode, x, u, dt)
    k1 = ode(x, u);
    k2 = ode(x + 0.5*dt*k1, u);
    k3 = ode(x + 0.5*dt*k2, u);
    k4 = ode(x + dt*k3, u);
    x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
end

% Nonlinear constraints (optional)
function [c, ceq] = nonlinear_constraints(u, x0, dt, x1_lb, x1_ub, x2_lb)
    N = length(u);           % Prediction horizon
    x = x0;                  % Initial state
    c = [];                  % Initialize inequality constraint array
    state_constraint = true;
    if state_constraint
        for k = 1:N
            % Simulate one step forward
            x = rk4(@(x, u) nonlinear_dynamics(x,u), x, u(k), dt);
    
            % Apply state constraints
            c_k = [ ...
                -x(1) + x1_lb;     % x1 >= x1_lb  →  -x1 + x1_lb <= 0
                 x(1) - x1_ub;     % x1 <= x1_ub   →   x1 - x1_ub <= 0
                -x(2) + x2_lb      % x2 >= x2_lb  →  -x2 + x2_lb <= 0
            ];
    
            c = [c; c_k];
        end
    end
    ceq = [];  % No equality constraints
end

% System dynamics function
function dxdt = nonlinear_dynamics(x, u)
    % Define system parameters
    mu = 1;

    % Extract state variables
    theta = x(1);
    theta_dot = x(2);
    
    % Define system dynamics
    theta_ddot = -mu*(theta^2 - 1)*theta_dot - theta + u;
    
    % Return the derivative of the state vector
    dxdt = [theta_dot; theta_ddot];
end

% Define the cost function for MPC
function cost = cost_function(u, x, x_ref, dt)
    % Define cost weights
    Q = diag([4, 1]); % state error weights
    Q_f = 2*Q;
    R = .1;          % control input weight
	N = length(u);
	x_traj = [x,zeros(2,N-1)];
	for i = 1:N-1
		x_traj(:,i+1) = rk4(@(x, u) nonlinear_dynamics(x_traj(:,i),u), x_traj(:,i), u(i), dt);
	end
    % Calculate state error
    state_error = x_traj(:) - repmat(x_ref, N, 1);
    
    % Construct the block diagonal cost matrix with Q and Q_f
    Q_blocks = kron(eye(N), Q);
    Q_blocks(end-size(Q,1)+1:end, end-size(Q,2)+1:end) = Q_f;
    
    % Calculate quadratic cost
    cost = state_error' * Q_blocks * state_error + u' * R * u;
end

function cost = lifted_cost_function_chopped(u, x, x_ref, dt)
    % Define cost weights
    Q = diag([4, 1]); % state error weights
    Q_f = 2*Q;
    R = .1;          % control input weight
	%N = length(u);
    N = 5;
    x_evol = x;

    % Some parameters for Simpson's 1/3 rule
    T = 10; % Sub-division factor
    sampling_div_2T = dt/(2 * T);
    sampling_div_T = dt/T;

    cost = 0;
	for i = 1:N
        for j = 1:T
            cintegral = (x_evol - x_ref).' * Q * (x_evol - x_ref);
            x_mid = rk4(@(x, u) nonlinear_dynamics(x_evol,u), x_evol, u((i-1)*T + j), sampling_div_2T);
            cintegral = cintegral + 4*((x_mid - x_ref).' * Q * (x_mid - x_ref));
            x_evol = rk4(@(x, u) nonlinear_dynamics(x_evol,u), x_evol, u((i-1)*T + j), sampling_div_T);
            cintegral = cintegral + (x_evol - x_ref).' * Q * (x_evol - x_ref);
            cost = cost + (1/6)*cintegral;
            cost = cost + (1/T)*R * u((i-1)*T + j)^2;
        end
    end
    terminal_pen = (x_evol - x_ref).' * Q_f * (x_evol - x_ref);
    cost = cost + terminal_pen;
end

function cost = lifted_cost_function(u, x, x_ref, dt)
    % Define cost weights
    Q = diag([4, 1]); % state error weights
    Q_f = 2*Q;
    R = .1;          % control input weight
	N = length(u);
    %N = 5;
    x_evol = x;

    % Some parameters for Simpson's 1/3 rule
    T = 10; % Sub-division factor
    sampling_div_2T = dt/(2 * T);
    sampling_div_T = dt/T;

    cost = 0;
	for i = 1:N
        for j = 1:T
            cintegral = (x_evol - x_ref).' * Q * (x_evol - x_ref);
            x_mid = rk4(@(x, u) nonlinear_dynamics(x_evol,u), x_evol, u(i), sampling_div_2T);
            cintegral = cintegral + 4*((x_mid - x_ref).' * Q * (x_mid - x_ref));
            x_evol = rk4(@(x, u) nonlinear_dynamics(x_evol,u), x_evol, u(i), sampling_div_T);
            cintegral = cintegral + (x_evol - x_ref).' * Q * (x_evol - x_ref);
            cost = cost + (1/6)*cintegral;
        end
        cost = cost + R * u(i)^2;
    end
    terminal_pen = (x_evol - x_ref).' * Q_f * (x_evol - x_ref);
    cost = cost + terminal_pen;
end