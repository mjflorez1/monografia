clc
clear
close all

% Parámetros iniciales
x_star = [0, 2, -3, 1];  % Solución real x* = (0,2,-3,1)
m = 46;  % Número total de datos

% Generación de t según la fórmula t_i = -1 + 0.1*i
i = (0:m-1)';  % Vector de índices de 0 a 45
t = -1 + 0.1 * i; % Generación correcta de t

% Generación de w según el modelo original sin ruido
w = x_star(1) + x_star(2) * t + x_star(3) * t.^2 + x_star(4) * t.^3;

% Introducir ruido aleatorio pequeño entre -0.01 y 0.01
rng(1);  % Fijar la semilla para reproducibilidad
r = (rand(m, 1) - 0.5) * 0.02;

% Generar y con ruido
y = w + r;

% Introducir valores atípicos en los índices 7 a 16 (corresponde a i = 7, ..., 16)
y(7:16) = 10;

% Ajuste con POLYFIT (ajuste polinómico de grado 3)
coeff_polyfit = polyfit(t, y, 3);

% Evaluar el polinomio ajustado en los puntos t
y_polyfit = polyval(coeff_polyfit, t);

% Mostrar coeficientes ajustados
disp('Coeficientes del polinomio ajustado con polyfit:')
disp(coeff_polyfit)

% Gráfica comparativa
figure;
scatter(t, y, 'filled'); hold on; % Datos con ruido y valores atípicos
plot(t, w, 'r', 'LineWidth', 1.5); % Modelo original
plot(t, y_polyfit, 'm--', 'LineWidth', 1.5); % Ajuste con polyfit
legend('Datos con ruido y valores atípicos', 'Modelo verdadero', 'Ajuste con polyfit');
xlabel('t');
ylabel('y');
title('Comparación del ajuste POLYFIT con los datos generados');
grid on;