%%writefile temp_change_mean_multistream_extended.cu
// ------------------------------------------------------------
// Promedio por fila usando CUDA con MÚLTIPLES STREAMS
// Modificado para ejecutar con 1, 2, 4 u 8 streams
// ------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define COL_START 5
#define COL_END 14
#define COLS (COL_END - COL_START + 1)
#define ROWS 8636
#define MAX_COUNTRY_NAME 100

// 1, 2, 4 u 8
#define NUM_STREAMS 8

__global__ void meanKernel(
    const float* y0, const float* y1, const float* y2, const float* y3, const float* y4,
    const float* y5, const float* y6, const float* y7, const float* y8, const float* y9,
    float* out, int N, int offset
){
    int i = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (i < N) {
        float s = y0[i] + y1[i] + y2[i] + y3[i] + y4[i]
                + y5[i] + y6[i] + y7[i] + y8[i] + y9[i];
        out[i] = s / 10.0f;
    }
}

int main(){
    // --------------------------------------------------------
    // 1) Leer CSV
    // --------------------------------------------------------
    FILE *f = fopen("Environment_Temperature_change_filled_extended.csv","r");
    if(!f){
        printf("No se pudo abrir CSV\n");
        return 0;
    }

    char line[5000];
    if(!fgets(line, 5000, f)) return 0;

    // --------------------------------------------------------
    // 2) Reservar memoria en host
    // --------------------------------------------------------
    float *y0,*y1,*y2,*y3,*y4,*y5,*y6,*y7,*y8,*y9;
    float *out_host;
    char **countries;

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
    
    countries = (char**)malloc(ROWS*sizeof(char*));
    for(int i = 0; i < ROWS; i++){
        countries[i] = (char*)malloc(MAX_COUNTRY_NAME*sizeof(char));
    }

    // --------------------------------------------------------
    // 3) Leer datos del CSV
    // --------------------------------------------------------
    int N = 0;
    while (N < ROWS && fgets(line, 5000, f)){
        char *tok = strtok(line, ",");
        int col = 0;
        int taken = 0;

        while(tok){
            if(col == 0){
                strncpy(countries[N], tok, MAX_COUNTRY_NAME-1);
                countries[N][MAX_COUNTRY_NAME-1] = '\0';
            }
            else if(col >= COL_START && col <= COL_END){
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
                if(taken==10) break;
            }
            col++;
            tok = strtok(NULL, ",");
        }
        N++;
    }
    fclose(f);

    // --------------------------------------------------------
    // 4) Reservar memoria en device
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
    // 5) Crear múltiples streams
    // --------------------------------------------------------
    cudaStream_t streams[NUM_STREAMS];
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamCreate(&streams[i]);
    }

    // Eventos para medir tiempos globales
    cudaEvent_t h2d_start, h2d_end, k_start, k_end, d2h_start, d2h_end;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_end);
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_end);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_end);

    // --------------------------------------------------------
    // 6) Dividir trabajo entre streams
    // --------------------------------------------------------
    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    int tpb = 256;

    // INICIO Host->Device
    cudaEventRecord(h2d_start, 0);
    
    for(int s = 0; s < NUM_STREAMS; s++){
        int offset = s * chunk_size;
        int size = (offset + chunk_size > N) ? (N - offset) : chunk_size;
        
        if(size <= 0) continue;

        // Copiar cada columna
        cudaMemcpyAsync(d_y0 + offset, y0 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y1 + offset, y1 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y2 + offset, y2 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y3 + offset, y3 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y4 + offset, y4 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y5 + offset, y5 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y6 + offset, y6 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y7 + offset, y7 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y8 + offset, y8 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_y9 + offset, y9 + offset, size*sizeof(float), cudaMemcpyHostToDevice, streams[s]);
    }
    
    cudaEventRecord(h2d_end, 0);

    // INICIO Kernel
    cudaEventRecord(k_start, 0);
    
    for(int s = 0; s < NUM_STREAMS; s++){
        int offset = s * chunk_size;
        int size = (offset + chunk_size > N) ? (N - offset) : chunk_size;
        
        if(size <= 0) continue;

        int blocks = (size + tpb - 1) / tpb;
        meanKernel<<<blocks, tpb, 0, streams[s]>>>(
            d_y0, d_y1, d_y2, d_y3, d_y4,
            d_y5, d_y6, d_y7, d_y8, d_y9,
            d_out, offset + size, offset
        );
    }
    
    cudaEventRecord(k_end, 0);

    // INICIO Device->Host
    cudaEventRecord(d2h_start, 0);
    
    for(int s = 0; s < NUM_STREAMS; s++){
        int offset = s * chunk_size;
        int size = (offset + chunk_size > N) ? (N - offset) : chunk_size;
        
        if(size <= 0) continue;

        cudaMemcpyAsync(out_host + offset, d_out + offset, size*sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
    }
    
    cudaEventRecord(d2h_end, 0);

    // sincronizar al final
    cudaDeviceSynchronize();

    // --------------------------------------------------------
    // 7) Medir tiempos
    // --------------------------------------------------------
    float t_h2d, t_k, t_d2h;
    cudaEventElapsedTime(&t_h2d, h2d_start, h2d_end);
    cudaEventElapsedTime(&t_k, k_start, k_end);
    cudaEventElapsedTime(&t_d2h, d2h_start, d2h_end);

    printf("========================================\n");
    printf("Número de Streams: %d\n", NUM_STREAMS);
    printf("Filas procesadas: %d\n", N);
    printf("Chunk size por stream: %d\n", chunk_size);
    printf("========================================\n");
    printf("Tiempo Host->Device: %.6f ms\n", t_h2d);
    printf("Tiempo Kernel: %.6f ms\n", t_k);
    printf("Tiempo Device->Host: %.6f ms\n", t_d2h);
    printf("Tiempo Total: %.6f ms\n", t_h2d + t_k + t_d2h);
    printf("========================================\n\n");

    // --------------------------------------------------------
    // 8) Encontrar país con promedio máximo
    // --------------------------------------------------------
    int max_idx = 0;
    float max_avg = out_host[0];
    
    for(int i = 1; i < N; i++){
        if(out_host[i] > max_avg){
            max_avg = out_host[i];
            max_idx = i;
        }
    }
    
    printf("País con promedio más alto: %s\n", countries[max_idx]);
    printf("Promedio: %.6f\n\n", max_avg);

    // --------------------------------------------------------
    // 9) Limpieza
    // --------------------------------------------------------
    for(int i = 0; i < NUM_STREAMS; i++){
        cudaStreamDestroy(streams[i]);
    }
    
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_end);
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_end);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_end);

    cudaFree(d_y0); cudaFree(d_y1); cudaFree(d_y2); cudaFree(d_y3); cudaFree(d_y4);
    cudaFree(d_y5); cudaFree(d_y6); cudaFree(d_y7); cudaFree(d_y8); cudaFree(d_y9);
    cudaFree(d_out);

    free(y0); free(y1); free(y2); free(y3); free(y4);
    free(y5); free(y6); free(y7); free(y8); free(y9);
    free(out_host);
    
    for(int i = 0; i < ROWS; i++){
        free(countries[i]);
    }
    free(countries);

    return 0;
}