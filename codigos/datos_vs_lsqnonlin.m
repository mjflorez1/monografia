clc
clear
close all

% Cargar los datos desde el archivo
datos = readmatrix('data.txt', 'NumHeaderLines', 1);  % Saltar la línea de encabezado

% Extraer columnas
t = datos(:,1);  % Primera columna (valores de t)
y = datos(:,2);  % Segunda columna (valores de y)

% Definir los coeficientes del modelo verdadero
x_star = [0, 2, -3, 1];  % Coeficientes reales (como en el código original)

% Calcular los valores de w (modelo verdadero sin ruido)
w = x_star(1) + x_star(2) * t + x_star(3) * t.^2 + x_star(4) * t.^3;

% AJUSTE CON LSQNONLIN 
% Punto inicial para el ajuste
x0 = [-1.0, -2.0, 1.0, -1.0];

% Definir la función polinómica a ajustar
modelo_polinomico = @(x, t) x(1) + x(2)*t + x(3)*t.^2 + x(4)*t.^3;

% Definir la función de error para lsqnonlin
fun = @(x) modelo_polinomico(x, t) - y;

% Configurar límites para los coeficientes
lb = [-10, -10, -10, -10]; % Límites inferiores
ub = [10, 10, 10, 10];     % Límites superiores

% Resolver el problema de mínimos cuadrados no lineales
options = optimoptions('lsqnonlin', 'Display', 'iter'); % Mostrar iteraciones
x_lsq = lsqnonlin(fun, x0, lb, ub, options);

% Evaluar el modelo ajustado con lsqnonlin
y_lsq = modelo_polinomico(x_lsq, t);

% Mostrar coeficientes obtenidos
disp('Coeficientes del polinomio ajustado con lsqnonlin:');
disp(x_lsq);

% GRÁFICA DEL AJUSTE 
figure;
scatter(t, y, 'filled'); hold on; % Datos con ruido y valores atípicos
plot(t, w, 'r', 'LineWidth', 1.5); % Modelo original
plot(t, y_lsq, 'g-.', 'LineWidth', 1.5); % Ajuste con lsqnonlin
legend('Datos con ruido y valores atípicos', 'Modelo verdadero', 'Ajuste con lsqnonlin');
xlabel('t');
ylabel('y');
title('Ajuste de Datos con LSQNONLIN');
grid on;