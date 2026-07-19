#include "ray.cuh"
#include "vect.cuh"
#include <iostream>
#include <time.h>
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
              << file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__global__ void render(vect *fb, int max_x, int max_y, point3 cameraCenter,
                       point3 vpFirstPixel, point3 vpHorizontalDel,
                       point3 vpVerticalDel) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  ray r(cameraCenter,
        vpFirstPixel + i * vpHorizontalDel + j * vpVerticalDel - cameraCenter);
  fb[pixel_index] = rayColor(r);
}

int main() {
  int tx = 32;
  int ty = 32;

  double aspectRatio = 16.0 / 9.0;
  int imgWidth = 1280;
  int imgHeight = max(1, static_cast<int>(imgWidth / aspectRatio));

  std::cerr << "Rendering a " << imgWidth << "x" << imgHeight << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  point3 cameraCenter = point3(0, 0, 0);
  double focalLen = 1.0;
  double vpHeight = 2.0;
  double vpWidth = vpHeight * (static_cast<double>(imgWidth) / imgHeight);

  vect vpHorizontal = vect(vpWidth, 0, 0);
  vect vpVertical = vect(0, -vpHeight, 0);

  vect vpHorizontalDel = vpHorizontal / imgWidth;
  vect vpVerticalDel = vpVertical / imgHeight;

  point3 vpCorner =
      cameraCenter + vect(0, 0, -focalLen) - vpHorizontal / 2 - vpVertical / 2;
  point3 vpFirstPixel = vpCorner + (vpHorizontalDel + vpVerticalDel) / 2;

  int num_pixels = imgHeight * imgWidth;
  size_t fb_size = num_pixels * sizeof(vect);

  // allocate FB
  vect *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks((imgWidth + tx - 1) / tx, (imgHeight + ty - 1 / ty) + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, imgWidth, imgHeight, cameraCenter,
                              vpFirstPixel, vpHorizontalDel, vpVerticalDel);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  freopen("out.ppm", "w", stdout);
  // Output FB as Image
  std::cout << "P3\n" << imgWidth << " " << imgHeight << "\n255\n";
  for (int j = 0; j < imgHeight; j++) {
    for (int i = 0; i < imgWidth; i++) {
      size_t pixel_index = j * imgWidth + i;

      float r = fb[pixel_index][0];
      float g = fb[pixel_index][1];
      float b = fb[pixel_index][2];

      int ir = int(255.99f * r);
      int ig = int(255.99f * g);
      int ib = int(255.99f * b);

      std::cout << ir << ' ' << ig << ' ' << ib << '\n';
    }
  }

  checkCudaErrors(cudaFree(fb));
}
