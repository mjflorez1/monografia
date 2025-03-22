clc; clear; close all;

% Cargar datos y extraer columnas
datos = readmatrix('data.txt', 'NumHeaderLines', 1);
t = datos(:,1); y = datos(:,2);

% Modelo verdadero sin ruido
x_star = [0, 2, -3, 1];
w = polyval(flip(x_star), t);

% Ajuste con LSQNONLIN
x0 = [-1, -2, 1, -1];
fun = @(x) polyval(flip(x), t) - y;
options = optimoptions('lsqnonlin', 'Display', 'iter');
x_lsq = lsqnonlin(fun, x0, -10*ones(1,4), 10*ones(1,4), options);

disp('Coeficientes del polinomio ajustado con lsqnonlin:');
disp(x_lsq);

% Ajuste con POLYFIT
x_poly = polyfit(t, y, 3);
disp('Coeficientes del polinomio ajustado con polyfit:');
disp(x_poly);

% Gráfica comparativa
figure;
scatter(t, y, 'filled'); hold on;
plot(t, w, 'r', 'LineWidth', 1.5);
plot(t, polyval(flip(x_lsq), t), 'g-.', 'LineWidth', 1.5);
plot(t, polyval(x_poly, t), 'b--', 'LineWidth', 1.5);
legend('Datos con ruido', 'Modelo verdadero', 'Ajuste con lsqnonlin', 'Ajuste con polyfit');
xlabel('t'); ylabel('y'); title('Comparación: LSQNONLIN vs POLYFIT'); grid on;