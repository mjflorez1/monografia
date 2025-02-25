clc
clear
close all

% Parametros iniciales
x_star = [0, 2, -3, 1];  % Solución verdadera
m = 46;  % Número de datos

% Generacion de datos
t = linspace(-1, 3.5, m); % Valores de t desde -1 hasta 3.5 con 46 puntos
w = x_star(1) + x_star(2)*t + x_star(3)*t.^2 + x_star(4)*t.^3; % Valores reales de la funcion

% Introducir ruido aleatorio pequeño
r = (rand(1, m) - 0.5) * 0.02;  % Ruido entre -0.01 y 0.01
y = w + r;  % Valores de salida con ruido

% Introducir valores atipicos en las posiciones 7 a 16
y(7:16) = 10;

% Parámetros iniciales
x_star = [0, 2, -3, 1];  % Solución verdadera
m = 46;  % Numero de datos

% Generacion de datos
t = linspace(-1, 3.5, m)'; % Valores de t en columna
w = x_star(1) + x_star(2)*t + x_star(3)*t.^2 + x_star(4)*t.^3; % Valores reales de la funcion

% Introducir ruido aleatorio pequeño
r = (rand(m, 1) - 0.5) * 0.02;  % Ruido entre -0.01 y 0.01
y = w + r;  % Valores de salida con ruido

% Introducir valores atipicos en las posiciones 7 a 16
y(7:16) = 10;

% Crear la tabla con los datos
data_table = table(t, y);
disp('Tabla 1: Datos generados para ajuste polinomico')
disp(data_table)

% Guardar la tabla en un archivo CSV (opcional)
writetable(data_table, 'datos_OVO.csv')

% Visualización de los datos
figure;
scatter(t, y, 'filled'); hold on;
plot(t, w, 'r', 'LineWidth', 1.5);
legend('Datos con ruido y valores atipicos', 'Modelo verdadero');
xlabel('t');
ylabel('y');
title('Generación de Datos para Ajuste Polinomico');
grid on;
