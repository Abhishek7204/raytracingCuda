#include "hitable_list.cuh"
#include "interval.cuh"
#include "ray.cuh"
#include "rt_utility.cuh"
#include "sphere.cuh"
#include "vect.cuh"
#include <float.h>
#include <iostream>
#include <time.h>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
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

__device__ vect rayColor(const ray &r, hitable **world) {
  hitRecord rec;
  if ((*world)->hit(r, interval(0, infinity), rec)) {
    return 0.5f * vect(rec.hitNormal.x() + 1.0f, rec.hitNormal.y() + 1.0f,
                       rec.hitNormal.z() + 1.0f);
  } else {
    vect unit_direction = unitVector(r.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    return (1.0f - t) * vect(1.0, 1.0, 1.0) + t * vect(0.5, 0.7, 1.0);
  }
}

__global__ void render(vect *fb, int max_x, int max_y, vect lower_left_corner,
                       vect horizontal, vect vertical, vect origin,
                       hitable **world) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u * horizontal + v * vertical);
  fb[pixel_index] = rayColor(r, world);
}

__global__ void create_world(hitable **d_list, hitable **d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vect(0, 0, -1), 0.5);
    *(d_list + 1) = new sphere(vect(0, -100.5, -1), 100);
    *d_world = new hitable_list(d_list, 2);
  }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
  delete *(d_list);
  delete *(d_list + 1);
  delete *d_world;
}

int main() {
  double aspectRatio = 16.0 / 9.0;
  int nx = 1280;
  int ny = max(1, static_cast<int>(nx / aspectRatio));
  int tx = 16;
  int ty = 16;

  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  freopen("out.ppm", "w", stdout);

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vect);

  // allocate FB
  vect *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // make our world of hitables
  hitable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2 * sizeof(hitable *)));
  hitable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
  create_world<<<1, 1>>>(d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny, vect(-2.0, -1.0, -1.0),
                              vect(4.0, 0.0, 0.0), vect(0.0, 2.0, 0.0),
                              vect(0.0, 0.0, 0.0), d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      int ir = int(255.99 * fb[pixel_index].x());
      int ig = int(255.99 * fb[pixel_index].y());
      int ib = int(255.99 * fb[pixel_index].z());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list, d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}
