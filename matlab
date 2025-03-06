clc;
clear;
close all;

% Define the 4R robot using DH parameters
L1 = Link('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [0.5, 0, 0], 'I', [0.1, 0.1, 0.1], 'Jm', 0.01);
L2 = Link('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [0.5, 0, 0], 'I', [0.1, 0.1, 0.1], 'Jm', 0.01);
L3 = Link('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [0.5, 0, 0], 'I', [0.1, 0.1, 0.1], 'Jm', 0.01);
L4 = Link('d', 0, 'a', 1, 'alpha', 0, 'm', 1, 'r', [0.5, 0, 0], 'I', [0.1, 0.1, 0.1], 'Jm', 0.01);

robot = SerialLink([L1 L2 L3 L4], 'name', '4R Robot');

% Define time parameters
dt = 0.1;        
steps = 450;      

% Define the circular trajectory parameters
center_x = 2;     
center_y = 2;     
radius = 1;       

% Generate the desired circular trajectory
theta = linspace(0, 2 * pi, steps);  
x_d = center_x + radius * cos(theta); 
y_d = center_y + radius * sin(theta); 

% Compute desired velocities and accelerations
x_dot_d = gradient(x_d, dt);
y_dot_d = gradient(y_d, dt);
x_ddot_d = gradient(x_dot_d, dt);
y_ddot_d = gradient(y_dot_d, dt);

% Symbolic variables for time-dependent joint angles
syms q1(t) q2(t) q3(t) q4(t) real         
syms q1_dot(t) q2_dot(t) q3_dot(t) q4_dot(t) real 

q_sym = [q1(t); q2(t); q3(t); q4(t)];
q_dot_sym = [q1_dot(t); q2_dot(t); q3_dot(t); q4_dot(t)];

% Define forward kinematics symbolically
x_end = cos(q1(t)) + cos(q1(t) + q2(t)) + cos(q1(t) + q2(t) + q3(t)) + cos(q1(t) + q2(t) + q3(t) + q4(t));
y_end = sin(q1(t)) + sin(q1(t) + q2(t)) + sin(q1(t) + q2(t) + q3(t)) + sin(q1(t) + q2(t) + q3(t) + q4(t));
pos_end = [x_end; y_end];

% Compute the Jacobian symbolically
J_sym = jacobian(pos_end, q_sym);

% Compute the time derivative of the Jacobian symbolically
J_dot_sym = diff(J_sym, t);

% Substitute angular velocities into J_dot
J_dot_sym = subs(J_dot_sym, ...
    [diff(q1(t), t), diff(q2(t), t), diff(q3(t), t), diff(q4(t), t)], ...
    [q1_dot(t), q2_dot(t), q3_dot(t), q4_dot(t)]);

% Initialize numerical values for simulation
q_vals = [0.1; 0.1; 0.1; 0.1];        % Initial joint angles
q_dot_vals = [0; 0; 0; 0];            % Initial joint velocities
q_ddot = [0; 0; 0; 0];                % Initial joint accelerations
trajectory = zeros(steps, 2);         % To record end-effector position

% Define the gravity Vector [XY Plane]
robot.gravity = [0; -9.81; 0];

% PD control gains
% Kp = 5;
% Kd = 1.4;
Kp = 1.7;
Kd = 0.8;

% Initialize variables for plotting
end_effector_vel = zeros(steps, 2);    % To store end-effector velocities
end_effector_acc = zeros(steps, 2);    % To store end-effector accelerations
observed_forces = zeros(steps, 2);     % To store observed forces
commanded_forces = zeros(steps, 2);    % To store commanded forces
force_errors = zeros(steps, 2);        % To store force errors
position_errors = zeros(steps, 2);     % To store position errors
velocity_errors = zeros(steps, 2);     % To store velocity errors
acceleration_errors = zeros(steps, 2); % To store acceleration errors

% Main control loop
for i = 1:steps

    % disp(i);

    % Current joint angles and velocities
    q1_val = q_vals(1);
    q2_val = q_vals(2);
    q3_val = q_vals(3);
    q4_val = q_vals(4);
    q1_dot_val = q_dot_vals(1);
    q2_dot_val = q_dot_vals(2);
    q3_dot_val = q_dot_vals(3);
    q4_dot_val = q_dot_vals(4);

    disp(q_vals);
    disp(q_dot_vals);

    % Evaluate Jacobian and Jacobian derivative numerically
    J = double(subs(J_sym, [q1(t), q2(t), q3(t), q4(t)], [q1_val, q2_val, q3_val, q4_val]));
    J_dot = double(subs(J_dot_sym, ...
        [q1(t), q2(t), q3(t), q4(t), q1_dot(t), q2_dot(t), q3_dot(t), q4_dot(t)], ...
        [q1_val, q2_val, q3_val, q4_val, q1_dot_val, q2_dot_val, q3_dot_val, q4_dot_val]));

    % Compute current end-effector position
    x_observed = double(subs(x_end, [q1(t), q2(t), q3(t), q4(t)], [q1_val, q2_val, q3_val, q4_val]));
    y_observed = double(subs(y_end, [q1(t), q2(t), q3(t), q4(t)], [q1_val, q2_val, q3_val, q4_val]));
    trajectory(i, :) = [x_observed, y_observed];

    % Compute end-effector velocity and acceleration
    end_effector_vel(i, :) = (J * [q1_dot_val; q2_dot_val; q3_dot_val; q4_dot_val])';  % Velocity
    end_effector_acc(i, :) = (J * q_ddot + J_dot * [q1_dot_val; q2_dot_val; q3_dot_val; q4_dot_val])';  % Acceleration

    % Compute position and velocity errors
    e = [x_d(i) - x_observed; y_d(i) - y_observed];
    e_dot = [x_dot_d(i); y_dot_d(i)] - J * q_dot_vals;
    e_ddot = [x_ddot_d(i); y_ddot_d(i)] - J_dot * q_dot_vals - J * q_ddot;
    position_errors(i, :) = e';  % Store position error
    velocity_errors(i, :) = e_dot';  % Store velocity error
    acceleration_errors(i, :) = [x_ddot_d(i) - end_effector_acc(i, 1), y_ddot_d(i) - end_effector_acc(i, 2)];

    % PD control for joint accelerations
    q_ddot = pinv(J) * (Kp * e + Kd * e_dot + e_ddot);

    % Compute dynamics (inertia, Coriolis, gravity)
    B = robot.inertia(q_vals');
    C = robot.coriolis(q_vals', q_dot_vals');
    G = robot.gravload(q_vals');

    % disp("Gravity: ");
    % disp(G);

    % Compute joint torques
    tau = B * q_ddot + C * q_dot_vals + G';

    % disp(tau(1));
    % disp(tau(2));
    % disp(tau(3));
    % disp(tau(4));

    % Compute observed forces
    F_observed = J * tau;
    observed_forces(i, :) = F_observed';  % Store observed forces

    % Compute commandded end effector forces;

    q_dot_desired = pinv(J) * [x_dot_d(i); y_dot_d(i)];
    q_ddot_desired = pinv(J) * ([x_ddot_d(i); y_ddot_d(i)] - J_dot * q_dot_vals);

    tau_commanded = B * q_ddot_desired + C * q_dot_desired + G;

    F_commanded = J * tau_commanded;

    F_commanded = F_commanded(1:2);

    commanded_forces(i, :) = F_commanded;

    % Compute force errors
    force_errors(i, :) = F_commanded - F_observed';

    % Simulate joint motion using ODE45
    tspan = [0 dt];
    [~, y_ode] = ode45(@(t, y) robot_dynamics(t, y, tau, robot), tspan, [q_vals; q_dot_vals]);

    % Update joint states
    q_vals = y_ode(end, 1:4)';
    q_dot_vals = y_ode(end, 5:8)';

    % Plot the robot and trajectory in real-time
    robot.plot(q_vals');
    hold on;
    plot(trajectory(1:i, 1), trajectory(1:i, 2), 'r');  % Actual trajectory
    plot(x_d, y_d, 'b--');                              % Desired trajectory
    drawnow;

    % disp(i);

end

% ODE Helper Function
function dydt = robot_dynamics(~, y, tau, robot)
    q = y(1:4);          % Joint positions
    q_dot = y(5:8);      % Joint velocities

    % Compute dynamics
    B = robot.inertia(q');
    C = robot.coriolis(q', q_dot');
    G = robot.gravload(q');

    % Joint accelerations
    q_ddot = B \ (tau - C * q_dot - G');

    dydt = [q_dot; q_ddot];
end

% Plot end-effector position, velocity, and acceleration
figure;

% Position Plot
subplot(2,2,1);
plot((0:steps-1) * dt, trajectory(:, 1), 'r', 'LineWidth', 1.5); % Actual X Position
hold on;
plot((0:steps-1) * dt, trajectory(:, 2), 'k', 'LineWidth', 1.5); % Actual Y Position
plot((0:steps-1) * dt, x_d, 'g--', 'LineWidth', 1.5);            % Desired X Position
plot((0:steps-1) * dt, y_d, 'b--', 'LineWidth', 1.5);            % Desired Y Position
xlabel('Time (s)');
ylabel('Position (m)');
legend('X Position (Actual)', 'Y Position (Actual)', 'X Position (Desired)', 'Y Position (Desired)');
title('End-Effector Position');
grid on;

% Velocity Plot
subplot(2,2,2);
plot((0:steps-1) * dt, end_effector_vel(:, 1), 'r', 'LineWidth', 1.5); % Actual X Velocity
hold on;
plot((0:steps-1) * dt, end_effector_vel(:, 2), 'k', 'LineWidth', 1.5); % Actual Y Velocity
plot((0:steps-1) * dt, x_dot_d, 'g--', 'LineWidth', 1.5);              % Desired X Velocity
plot((0:steps-1) * dt, y_dot_d, 'b--', 'LineWidth', 1.5);              % Desired Y Velocity
xlabel('Time (s)');
ylabel('Velocity (m/s)');
ylim([-1,1]);
legend('X Velocity (Actual)', 'Y Velocity (Actual)', 'X Velocity (Desired)', 'Y Velocity (Desired)');
title('End-Effector Velocity');
grid on;

% Acceleration Plot
subplot(2,2,3);
plot((0:steps-1) * dt, end_effector_acc(:, 1), 'r', 'LineWidth', 1.5); % Actual X Acceleration
hold on;
plot((0:steps-1) * dt, end_effector_acc(:, 2), 'k', 'LineWidth', 1.5); % Actual Y Acceleration
plot((0:steps-1) * dt, x_ddot_d, 'g--', 'LineWidth', 1.5);             % Desired X Acceleration
plot((0:steps-1) * dt, y_ddot_d, 'b--', 'LineWidth', 1.5);             % Desired Y Acceleration
xlabel('Time (s)');
ylabel('Acceleration (m/s^2)');
ylim([-0.5,0.5]);
legend('X Acceleration (Actual)', 'Y Acceleration (Actual)', 'X Acceleration (Desired)', 'Y Acceleration (Desired)');
title('End-Effector Acceleration');
grid on;

% Commanded and Observed Forces
subplot(2, 2, 4);
plot((0:steps-1) * dt, observed_forces(:, 1), 'r', 'LineWidth', 1.5);  % Observed X Force
hold on;
plot((0:steps-1) * dt, observed_forces(:, 2), 'k', 'LineWidth', 1.5);  % Observed Y Force
plot((0:steps-1) * dt, commanded_forces(:, 1), 'g--', 'LineWidth', 1.5); % Commanded X Force
plot((0:steps-1) * dt, commanded_forces(:, 2), 'b--', 'LineWidth', 1.5); % Commanded Y Force
xlabel('Time (s)');
ylabel('Force (N)');
ylim([-5, 5]);
legend('Observed X Force', 'Observed Y Force', 'Commanded X Force', 'Commanded Y Force');
title('End-Effector Forces');
grid on;

% Plot errors for position, velocity, and acceleration
figure;

% Position Error Plot
subplot(2,2,1);
plot((0:steps-1) * dt, position_errors(:, 1), 'r', 'LineWidth', 1.5); % X Position Error
hold on;
plot((0:steps-1) * dt, position_errors(:, 2), 'k', 'LineWidth', 1.5); % Y Position Error
xlabel('Time (s)');
ylabel('Position Error (m)');
legend('X Error', 'Y Error');
title('End-Effector Position Errors');
grid on;

% Velocity Error Plot
subplot(2,2,2);
plot((0:steps-1) * dt, velocity_errors(:, 1), 'r', 'LineWidth', 1.5); % X Velocity Error
hold on;
plot((0:steps-1) * dt, velocity_errors(:, 2), 'k', 'LineWidth', 1.5); % Y Velocity Error
xlabel('Time (s)');
ylabel('Velocity Error (m/s)');
ylim([-1,1]);
legend('X Error', 'Y Error');
title('End-Effector Velocity Errors');
grid on;

% Acceleration Error Plot
subplot(2,2,3);
plot((0:steps-1) * dt, acceleration_errors(:, 1), 'r', 'LineWidth', 1.5); % X Acceleration Error
hold on;
plot((0:steps-1) * dt, acceleration_errors(:, 2), 'k', 'LineWidth', 1.5); % Y Acceleration Error
xlabel('Time (s)');
ylabel('Acceleration Error (m/s^2)');
ylim([-0.5,0.5]);
legend('X Error', 'Y Error');
title('End-Effector Acceleration Errors');
grid on;

% Force Error Plot
subplot(2,2,4); 
plot((0:steps-1) * dt, force_errors(:, 1), 'r', 'LineWidth', 1.5); % X Force Error
hold on;
plot((0:steps-1) * dt, force_errors(:, 2), 'k', 'LineWidth', 1.5); % Y Force Error
xlabel('Time (s)');
ylabel('Force Error (N)');
ylim([-5,5]);
legend('X Force Error', 'Y Force Error');
title('End-Effector Force Errors');
grid on;