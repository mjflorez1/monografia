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

% Mostrar los coeficientes encontrados
disp('Coeficientes obtenidos con polyfit:');
disp(coef_polyfit);

% Evaluación del polinomio ajustado
y_fit = polyval(coef_polyfit, t);

% Graficar los resultados
figure;
plot(t, y, 'ro', 'MarkerSize', 6, 'DisplayName', 'Datos con ruido y outliers');
hold on;
plot(t, y_real, 'b-', 'LineWidth', 2, 'DisplayName', 'Valores reales');
plot(t, y_fit, 'g--', 'LineWidth', 2, 'DisplayName', 'Ajuste polyfit');
legend;
xlabel('t');
ylabel('y');
title('Ajuste polinómico con polyfit');
grid on;