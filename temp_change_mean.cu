%%writefile temp_change_mean.cu
// ------------------------------------------------------------
// Promedio por fila (columnas F..O) usando CUDA
// - Lee el CSV "Environment_Temperature_change_filled.csv"
// - Toma exactamente 10 columnas consecutivas (F..O)
// - Guarda cada columna en su propio vector (y0..y9)
// - Copia a GPU, ejecuta un kernel (1 hilo = 1 fila), calcula el
//   promedio por fila con suma directa, y trae el resultado a host
// - Usa UN SOLO stream y mide tiempos: H->D, Kernel, D->H
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Rango de columnas a usar (base 0):
// F = 5, ..., O = 14,  total 10 columnas
#define COL_START 5
#define COL_END   14
#define COLS      (COL_END - COL_START + 1)

// Filas reales de datos del archivo (4319 totales menos 1 del encabezado)
#define ROWS 4318

// ------------------------------------------------------------
// Kernel: un hilo procesa una fila
// Recibe 10 vectores (una columna por vector) y escribe en out[i]
// el promedio simple de esas 10 entradas.
// ------------------------------------------------------------
__global__ void meanKernel(
    const float* y0, const float* y1, const float* y2, const float* y3, const float* y4,
    const float* y5, const float* y6, const float* y7, const float* y8, const float* y9,
    float* out, int N
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Suma directa: aquí son 10 columnas fijas, más claro que un for
        float s = y0[i] + y1[i] + y2[i] + y3[i] + y4[i]
                + y5[i] + y6[i] + y7[i] + y8[i] + y9[i];
        out[i] = s / 10.0f;
    }
}

int main(){
    // --------------------------------------------------------
    // 1) Abrir CSV y descartar encabezado
    // --------------------------------------------------------
    FILE *f = fopen("Environment_Temperature_change_filled.csv","r");
    if(!f){
        printf("no se pudo abrir CSV\n");
        return 0;
    }

    // Buffer amplio por línea: depende del largo de la línea, no del # de filas
    char line[5000];

    // Leer y descartar la primera línea (header). Usamos el retorno para evitar warnings.
    if(!fgets(line, 5000, f)) return 0;

    // --------------------------------------------------------
    // 2) Reservar vectores por columna (F..O) y de salida
    //    Como el archivo es estático: reservamos ROWS exactas
    // --------------------------------------------------------
    float *y0,*y1,*y2,*y3,*y4,*y5,*y6,*y7,*y8,*y9;
    float *out_host;

    y0 = (float*)malloc(ROWS*sizeof(float));
    y1 = (float*)malloc(ROWS*sizeof(float));
    y2 = (float*)malloc(ROWS*sizeof(float));
    y3 = (float*)malloc(ROWS*sizeof(float));
    y4 = (float*)malloc(ROWS*sizeof(float));
    y5 = (float*)malloc(ROWS*sizeof(float));
    y6 = (float*)malloc(ROWS*sizeof(float));
    y7 = (float*)malloc(ROWS*sizeof(float));
    y8 = (float*)malloc(ROWS*sizeof(float));
    y9 = (float*)malloc(ROWS*sizeof(float));
    out_host = (float*)malloc(ROWS*sizeof(float));

    // --------------------------------------------------------
    // 3) Leer el CSV línea por línea y llenar y0..y9
    //    - strtok separa por comas
    //    - solo copiamos columnas [COL_START..COL_END]
    //    - atof convierte texto -> float (si no es numérico, queda 0.0)
    // --------------------------------------------------------
    int N = 0;  // filas efectivamente cargadas
    while (N < ROWS && fgets(line, 5000, f)){
        char *tok = strtok(line, ",");
        int col = 0;
        int taken = 0; // cuántas columnas de F..O llevamos (0..9)

        while(tok){
            if(col >= COL_START && col <= COL_END){
                float v = atof(tok);
                switch(taken){
                    case 0: y0[N]=v; break;
                    case 1: y1[N]=v; break;
                    case 2: y2[N]=v; break;
                    case 3: y3[N]=v; break;
                    case 4: y4[N]=v; break;
                    case 5: y5[N]=v; break;
                    case 6: y6[N]=v; break;
                    case 7: y7[N]=v; break;
                    case 8: y8[N]=v; break;
                    case 9: y9[N]=v; break;
                }
                taken++;
                if(taken==10) break; // ya tomamos F..O
            }
            col++;
            tok = strtok(NULL, ",");
        }
        N++;
    }
    fclose(f); // listo, CSV cargado a columnas

    // --------------------------------------------------------
    // 4) Reservar memoria en device (GPU)
    // --------------------------------------------------------
    float *d_y0,*d_y1,*d_y2,*d_y3,*d_y4,*d_y5,*d_y6,*d_y7,*d_y8,*d_y9,*d_out;
    cudaMalloc((void**)&d_y0, N*sizeof(float));
    cudaMalloc((void**)&d_y1, N*sizeof(float));
    cudaMalloc((void**)&d_y2, N*sizeof(float));
    cudaMalloc((void**)&d_y3, N*sizeof(float));
    cudaMalloc((void**)&d_y4, N*sizeof(float));
    cudaMalloc((void**)&d_y5, N*sizeof(float));
    cudaMalloc((void**)&d_y6, N*sizeof(float));
    cudaMalloc((void**)&d_y7, N*sizeof(float));
    cudaMalloc((void**)&d_y8, N*sizeof(float));
    cudaMalloc((void**)&d_y9, N*sizeof(float));
    cudaMalloc((void**)&d_out, N*sizeof(float));

    // --------------------------------------------------------
    // 5) Un único stream y eventos para medir tiempos por etapa
    // --------------------------------------------------------
    cudaStream_t s;
    cudaStreamCreate(&s);

    cudaEvent_t h2d_s,h2d_e,k_s,k_e,d2h_s,d2h_e;
    cudaEventCreate(&h2d_s); cudaEventCreate(&h2d_e);
    cudaEventCreate(&k_s);   cudaEventCreate(&k_e);
    cudaEventCreate(&d2h_s); cudaEventCreate(&d2h_e);

    // --------------------------------------------------------
    // 6) Copia Host->Device (todas las columnas). Tiempo H->D.
    // --------------------------------------------------------
    cudaEventRecord(h2d_s, s);
    cudaMemcpyAsync(d_y0,y0,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y1,y1,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y2,y2,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y3,y3,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y4,y4,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y5,y5,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y6,y6,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y7,y7,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y8,y8,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaMemcpyAsync(d_y9,y9,N*sizeof(float),cudaMemcpyHostToDevice,s);
    cudaEventRecord(h2d_e, s);

    // --------------------------------------------------------
    // 7) Configuración de grid y lanzamiento del kernel. Tiempo kernel.
    //    - tpb=256 es un buen punto de partida
    //    - blocks = ceil(N/256)
    // --------------------------------------------------------
    int tpb = 256;
    int blocks = (N + tpb - 1) / tpb;

    cudaEventRecord(k_s, s);
    meanKernel<<<blocks, tpb, 0, s>>>(d_y0,d_y1,d_y2,d_y3,d_y4,
                                      d_y5,d_y6,d_y7,d_y8,d_y9,
                                      d_out, N);
    cudaEventRecord(k_e, s);

    // --------------------------------------------------------
    // 8) Copia Device->Host del vector de promedios. Tiempo D->H.
    // --------------------------------------------------------
    cudaEventRecord(d2h_s, s);
    cudaMemcpyAsync(out_host, d_out, N*sizeof(float), cudaMemcpyDeviceToHost, s);
    cudaEventRecord(d2h_e, s);

    // Aseguramos que todo terminó antes de leer tiempos o imprimir
   // cudaStreamSynchronize(s);

    // --------------------------------------------------------
    // 9) Tiempos (ms) por etapa y primeros resultados
    // --------------------------------------------------------
    float t_h2d, t_k, t_d2h;
    cudaEventElapsedTime(&t_h2d, h2d_s, h2d_e);
    cudaEventElapsedTime(&t_k,   k_s,   k_e);
    cudaEventElapsedTime(&t_d2h, d2h_s, d2h_e);

    printf("Filas procesadas: %d\n", N);
    printf("Tiempo Host->Device: %f ms\n", t_h2d);
    printf("Tiempo Kernel: %f ms\n", t_k);
    printf("Tiempo Device->Host: %f ms\n", t_d2h);

    // Vista rápida de los primeros 20 promedios para verificar que todo tiene sentido
    for(int i=0; i<20 && i<N; ++i){
        printf("Prom fila %d = %f\n", i, out_host[i]);
    }

    // --------------------------------------------------------
    // 10) Limpieza (host y device)
    // --------------------------------------------------------
    cudaFree(d_y0); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_y3); cudaFree(d_y4);
    cudaFree(d_y5); cudaFree(d_y6); cudaFree(d_y7); cudaFree(d_y8); cudaFree(d_y9);
    cudaFree(d_out);

    free(y0); free(y1); free(y2); free(y3); free(y4);
    free(y5); free(y6); free(y7); free(y8); free(y9);
    free(out_host);

    return 0;
}