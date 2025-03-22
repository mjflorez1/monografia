clc
clear all
close all

% Definición del modelo cubico
function result = model(t, x1, x2, x3, x4)
    result = x1 + (x2 * t) + (x3 * t^2) + (x4 * t^3);
end

function result = f_i(ti, yi, x)
    result = (model(ti, x(1), x(2), x(3), x(4)) - yi).^2;
end

% Número de datos
m = 46;
t = linspace(-1, 3.5, m);

% "Solución exacta"
xstar = [0 2 -3 1];

% Generación de datos
y = 10 * ones(m);
rng(1234); %Semilla
noise = 0.01;

for i = 1:6;
    y(i) = model(t(i), xstar(1), xstar(2), xstar(3), xstar(4)) + (-noise + 2 * noise*rand());
end

for i = 16:m;
    y(i) = model(t(i), xstar(1), xstar(2), xstar(3), xstar(4)) + (-noise + 2 * noise*rand());
end

% Punto inicial
xk = [-1 -2 1 -1];

%Funcion auxiliar
faux = zeros(1,m);

for i = 1:m
    faux(i) = f_i(t(i), y(i), xk);
end

org_idx = 1:m;
y_idx = 1:m;

[fsort, indices] = sort(faux);

p = 36;
eps = 0.01;
Ieps = [];

fovo = fsort(p);

for i = 1:m
    if abs(fovo - fsort(i)) < eps
        Ieps = [Ieps, indices];
    end
end

disp(Ieps)