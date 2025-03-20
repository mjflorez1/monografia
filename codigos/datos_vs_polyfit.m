clc
clear
close all

% Cargar los datos desde el archivo
datos = readmatrix('data.txt', 'NumHeaderLines', 1);  % Saltar la línea de encabezado

% Extraer columnas
t = datos(:,1);  % Primera columna (valores de t)
y = datos(:,2);  % Segunda columna (valores de y)

% Definir los coeficientes del modelo verdadero
x_star = [0, 2, -3, 1];  % Coeficientes reales

% Calcular los valores de w (modelo verdadero sin ruido)
w = x_star(1) + x_star(2) * t + x_star(3) * t.^2 + x_star(4) * t.^3;

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
