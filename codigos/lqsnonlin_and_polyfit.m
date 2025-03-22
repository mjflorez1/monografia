clc
clear all
close all

% Generación de datos
m = 46;
x_star = [0, 2, -3, 1]; % Solución original
t = linspace(-1, 3.5, m); % Valores de t
y_real = polyval(x_star, t); % Valores reales sin ruido

% Introducir ruido y valores atípicos
y = y_real + (rand(1, m) * 0.02 - 0.01); % Añadir ruido pequeño
y(7:16) = 10; % Introducir valores atípicos

% Ajuste polinómico con polyfit
grado = 3;
coef_polyfit = polyfit(t, y, grado);

% Mostrar los coeficientes obtenidos con polyfit
disp('Coeficientes obtenidos con polyfit:');
disp(coef_polyfit);

% Ajuste con lsqnonlin
x0 = [-1, -2, 1, -1];  % Valores iniciales para la optimización
lb = [-10, -10, -10, -10];  % Límites inferiores
ub = [10, 10, 10, 10];  % Límites superiores

% Función de error a minimizar
error_fun = @(x) (polyval(x, t) - y).^2;

% Aplicación de lsqnonlin
options = optimoptions('lsqnonlin', 'Display', 'iter');  % Opciones para mostrar iteraciones
coef_lsqnonlin = lsqnonlin(error_fun, x0, lb, ub, options);

% Mostrar los coeficientes obtenidos con lsqnonlin
disp('Coeficientes obtenidos con lsqnonlin:');
disp(coef_lsqnonlin);

% Evaluación de los modelos ajustados
y_fit_polyfit = polyval(coef_polyfit, t);
y_fit_lsqnonlin = polyval(coef_lsqnonlin, t);

% Graficar los resultados
figure;
plot(t, y, 'ro', 'MarkerSize', 6, 'DisplayName', 'Datos con ruido y outliers');
hold on;
plot(t, y_real, 'b-', 'LineWidth', 2, 'DisplayName', 'Valores reales');
plot(t, y_fit_polyfit, 'g--', 'LineWidth', 2, 'DisplayName', 'Ajuste polyfit');
plot(t, y_fit_lsqnonlin, 'm-.', 'LineWidth', 2, 'DisplayName', 'Ajuste lsqnonlin');
legend;
xlabel('t');
ylabel('y');
title('Comparación de Ajustes: polyfit vs lsqnonlin');
grid on;