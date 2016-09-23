% ----------------------------------------------------------------------
% Teor�a de Detecci�n y Estimaci�n - FIUBA
%
% Trabajo Final - 2014
%
% Alumno: Sebasti�n Sampayo
% ----------------------------------------------------------------------

% --- Funci�n estimadora por Kn vecinos m�s cercanos ---
% 
% [x, p] = knn_estimate(data, k, n) 
%
% p: funci�n de densidad de probabilidad estimada
% x: espacio creado donde se eval�a 'p'. (p = p(x))
% data: muestras de entrenamiento
% k: cantidad de vecinos m�s cercanos
% n: cantidad de muestras del espacio creado para 'p' (length(p)~n)
% 
% Ejemplo:
% data = rand(1e4,1);
% [x, p] = knn_estimate(data, 3, 1e5);
% plot(x,p)

function [x, p] = knn_estimate(data, k, n)
    x = linspace(min(data), max(data), n)';
    N = length(data);
    reshape(data, N,1);
    
    if k>1
        [IDXdata, Dx] = knnsearch(data, x, 'K', k);
        D = max(...
            abs(max(data(IDXdata),[],2)-min(data(IDXdata),[],2)),...
            max(Dx,[],2)...
            );
    else
        [~, D] = knnsearch(data, x);
    end
    p = k./(N*D);
end