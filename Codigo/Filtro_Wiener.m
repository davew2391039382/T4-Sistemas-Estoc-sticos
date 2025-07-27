% UNIVERSIDAD DE CUENCA
% TRABAJO 4 - PROCESOS ESTOCÁSTICOS - FILTRO WIENER
% ALEX PACHECO, DAVID MINCHALA, ANDRES CALDERÓN

clc; clear; close all;  % Limpiar consola, variables y cerrar figuras anteriores
%%  ***************************** GENERAR RUIDO  *****************************
function ruido = gen_ruido(img, sigma)      % Genera ruido Gaussiano con desviación estándar sigma y del mismo tamaño que la imagen 
    ruido = sigma * randn(size(img));       % randn genera valores con distribución normal (media 0, varianza 1)
end
%%  ******************************** BLURRING  *******************************
function [img_blur, psf] = blurr(img, kernel_size)
   psf = fspecial('motion', kernel_size, 60);  % 60° para efecto motion más marcado  % PSF Gaussiana simétrica
    img_blur = imfilter(img, psf, 'conv', 'circular'); % Evita discontinuidades en bordes y aplica el blur a la imagen
end
%%  ****************************** FILTRO PROPIO *****************************
function i_rest = f_propio(g, H, Pn, Ps)    % Implementación manual del filtro de Wiener en el dominio de la frecuencia
    G = fft2(g);                            % Transformada de Fourier de la imagen degradada (blur + ruido)
    Hf = fft2(H, size(g, 1), size(g, 2));   % Transformada del kernel PSF del mismo tamaño que G
    H_conj = conj(Hf);                      % Conjugado complejo de Hf
    epsilon = 1e-3;                         % Pequeño valor para evitar división por cero
    W = H_conj ./ (abs(Hf).^2 + (Pn / (Ps + epsilon)));  % Filtro de Wiener en el dominio de la frecuencia
    F_hat = W .* G;                         % Aplica el filtro al espectro degradado
    i_rest = real(ifft2(F_hat));            % Vuelve al dominio espacial con la parte real
end
%%  *********************** CARGA Y NORMALIZACION *****************************
i_orig = imread('espacio_gris.png');              % Cargar imagen original
if size(i_orig, 3) == 3                     % Si es RGB (3 canales)...
    i_orig = rgb2gray(i_orig);              % Convertir a escala de grises
end
i_orig = im2double(i_orig);                 % Normalizar imagen a rango [0,1] (tipo double)
%%  ******************************* PARAMETROS  *******************************
kernel_size = 40;                           % Tamaño del kernel para desenfoque (PSF)
SNR_dB = 10;                                % Relación señal-ruido deseada en decibelios

%%  ****************************** PROCESAMIENTO  *****************************
[img_blur, psf] = blurr(i_orig, kernel_size);  

mu = mean(i_orig(:));
Ps_e = mean(i_orig(:).^2);           % Potencia total
Ps = Ps_e - mu^2;          % Varianza = potencia - media^2% Aplica blurring y obtiene la PSF usada                                 % Calcula la potencia de la señal como varianza de la imagen original
Pn = Ps / (10^(SNR_dB / 10));                           % Calcula la potencia del ruido a partir del SNR en dB
sigma = sqrt(Pn);                                       % Desviación estándar del ruido Gaussiano
ruido = gen_ruido(i_orig, sigma);                       % Genera ruido Gaussiano puro
i_orig_ruido = i_orig + ruido;                          % Suma ruido a la imagen original (sin blur)
i_bluruid = blurr(i_orig_ruido, kernel_size);           % Aplica blurring sobre la imagen con ruido
i_rest_manual = f_propio(i_bluruid, psf, Pn, Ps);       % Restaura la imagen usando filtro de Wiener manual
i_rest_manual = min(max(i_rest_manual, 0), 1);          % Recorta valores fuera de [0,1]
i_rest_matlab = deconvwnr(i_bluruid, psf, Pn / Ps);     % Aplica el filtro de Wiener de MATLAB
i_rest_matlab = min(max(i_rest_matlab, 0), 1);          % Recorta valores fuera de [0,1]
mse_manual = immse(i_orig, i_rest_manual);              % Calcula el error cuadrático medio de la restauración manual
mse_matlab = immse(i_orig, i_rest_matlab);              % Calcula el MSE de la restauración con función de MATLAB
%%  ******************************** GRAFICAS  *******************************
figure('Name', 'Filtro de Wiener - Comparaciones (Normalizado)', 'Position', [100, 100, 1400, 600]);  % Crea figura personalizada
subplot(2,4,1),imshow(i_orig),title('Imagen Original'); 
subplot(2,4,2),imshow(img_blur),title('Con Blurring');
subplot(2,4,3),imshow(i_orig_ruido),title('Original + Ruido');
subplot(2,4,4),imshow(i_bluruid),title('Blur + Ruido');
subplot(2,4,5),imshow(ruido, []),title('Ruido Gausiano');
subplot(2,4,6),imshow(i_rest_manual);   % Imagen restaurada con función Propia
title({sprintf('Restaurada (Manual)'), sprintf('MSE = %.6f', mse_manual)});  % Muestra MSE en el título
subplot(2,4,7), imshow(i_rest_matlab);  % Imagen restaurada con función de MATLAB
title({sprintf('Restaurada (MATLAB)'), sprintf('MSE = %.6f', mse_matlab)});  % Muestra MSE en el título
%% *************************** REPORTE EN CONSOLA  ***************************
fprintf('\n--- Resultados ---\n');                  % Imprime encabezado de resultados
fprintf('SNR objetivo: %.2f dB\n', SNR_dB);         % Imprime el SNR deseado
fprintf('Potencia señal (Ps): %.6f\n', Ps);         % Imprime potencia de la señal
fprintf('Potencia ruido (Pn): %.8f\n', Pn);         % Imprime potencia del ruido
fprintf('MSE (Wiener Manual): %.6f\n', mse_manual); % Muestra el MSE del filtro manual
fprintf('MSE (deconvwnr): %.6f\n', mse_matlab);     % Muestra el MSE del filtro MATLAB

