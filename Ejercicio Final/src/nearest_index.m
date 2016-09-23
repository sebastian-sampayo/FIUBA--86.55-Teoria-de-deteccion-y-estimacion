% ----------------------------------------------------------------------
% Teor�a de Detecci�n y Estimaci�n - FIUBA
%
% Trabajo Final - 2014
%
% Alumno: Sebasti�n Sampayo
% ----------------------------------------------------------------------

% - Funci�n para encontrar el �ndice del valor m�s cercano en un vector -
% Recibe un vector de entrada 'x', y otro vector de valores, 'values'.
% x_index: devuelve los �ndices del vector 'x' correspondientes
%   a los valores de 'x' m�s cercanos a los contenidos en 'values'
% 'values' puede ser un escalar o un vector.

function [x_index] = nearest_index(x, values)
    Nx = length(x);
    Nvalues = length(values);
    % Llevar los vectores a la dimensi�n requerida por el algoritmo
    % (columnas)
    x = reshape(x,Nx,1);
    values = reshape(values,Nvalues,1);
    % Calcular la distancia m�nima entre cada valor y 'x'
    d = abs(x*ones(1,Nvalues) - ones(Nx,1)*values');
    [~, x_index] = min(d);
end