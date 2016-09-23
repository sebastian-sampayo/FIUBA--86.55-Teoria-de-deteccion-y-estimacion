% ----------------------------------------------------------------------
% Teoría de Detección y Estimación - FIUBA
%
% Trabajo Final - 2014
%
% Alumno: Sebastián Sampayo
% ----------------------------------------------------------------------

% --- Función de Clasificación KNNR entre 2 clases ---
% 
% class = knnr_classificate(data, X0, X1, k)
%
% data: Muestras a clasificar (puede ser un escalar o un vector)
% X0: Muestras de entrenamiento de la clase 0
% X1: Muestras de entrenamiento de la clase 1
% class: Dato clasificado, 
%       class = 0 si clasifica clase 0 (de los k vecinos, mayoría de X0)
%       class = 1 si clasifica clase 1 (de los k vecinos, mayoría de X1)
% 
% Ejemplo:
% X0 = normrnd(2,1,1e3,1);
% X1 = rand(1e3,1)*10;
% knnr_classificate(rand*10,X0,X1,51)

function class = knnr_classificate(data, X0, X1, k)
    N0 = length(X0);
    N1 = length(X1);
    reshape(X0, N0,1);
    reshape(X1, N1,1);
    
    IDX = knnsearch([X0;X1], data, 'K', k);
    class = sum(IDX>N0,2) > k/2;
end