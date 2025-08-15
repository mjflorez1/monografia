clc
clear
close all

% Cargar los datos desde el archivo
datos = readmatrix('data.txt', 'NumHeaderLines', 1);  % Saltar la línea de encabezado

% Extraer columnas
t = datos(:,1);  % Primera columna (valores de t)
y = datos(:,2);  % Segunda columna (valores de y)

% Mostrar la tabla con los datos cargados
data_table = table(t, y);
disp('Tabla: Datos cargados desde data.txt')
disp(data_table)

% Graficar los datos cargados
figure;
plot(t, y, 'bo-', 'MarkerFaceColor', 'b');
xlabel('t');
ylabel('y');
title('Datos cargados desde data.txt');
grid on;