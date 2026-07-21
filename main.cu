#include "camera.cuh"
#include "curand_kernel.h"
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

#define RANDVECt                                                               \
  vect(curand_uniform(local_rand_state), curand_uniform(local_rand_state),     \
       curand_uniform(local_rand_state))

__device__ vect random_in_unit_sphere(curandState *local_rand_state) {
  vect p;
  do {
    p = 2.0f * RANDVECt - vect(1, 1, 1);
  } while (p.lenSquared() >= 1.0f);
  return p;
}

__device__ vect rayColor(const ray &r, int depthLeft, hitable **world,
                         curandState *local_rand_state) {
  if (!depthLeft)
    return color();
  ray cur_ray = r;
  float cur_attenuation = 1.0f;
  for (int i = 0; i < 50; i++) {
    hitRecord rec;
    if ((*world)->hit(cur_ray, interval(0, infinity), rec)) {
      vect target = rec.contactPoint + rec.hitNormal +
                    random_in_unit_sphere(local_rand_state);
      cur_attenuation *= 0.5f;
      cur_ray = ray(rec.contactPoint, target - rec.contactPoint);
    } else {
      vect unit_direction = unitVector(cur_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      vect c = (1.0f - t) * vect(1.0, 1.0, 1.0) + t * vect(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return vect(0, 0, 0);
}

__global__ void render(vect *fb, int max_x, int max_y, int sampleCount,
                       int sampleDepth, camera **cam, hitable **world) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y))
    return;
  int pixel_index = j * max_x + i;
  curandState state;
  curand_init(1984, pixel_index, 0, &state);

  color sum(0, 0, 0);
  for (int it = 0; it < sampleCount; it++) {
    double u = double(i + curand_uniform(&state)) / double(max_x);
    double v = double(j + curand_uniform(&state)) / double(max_y);
    ray r = (*cam)->get_ray(u, v);
    sum += rayColor(r, sampleDepth, world, &state);
  }
  sum = sum / sampleCount;
  sum[0] = sqrt(sum[0]);
  sum[1] = sqrt(sum[1]);
  sum[2] = sqrt(sum[2]);
  fb[pixel_index] = sum;
}

__global__ void create_world(hitable **d_list, hitable **d_world,
                             camera **d_camera) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list) = new sphere(vect(0, 0, -1), 0.5);
    *(d_list + 1) = new sphere(vect(0, -100.5, -1), 100);
    *d_world = new hitable_list(d_list, 2);
    *d_camera = new camera();
  }
}

__global__ void free_world(hitable **d_list, hitable **d_world,
                           camera **d_camera) {
  delete *(d_list);
  delete *(d_list + 1);
  delete *d_world;
  delete *d_camera;
}

int main() {
  double aspectRatio = 16.0 / 9.0;
  int nx = 1280;
  int ny = max(1, static_cast<int>(nx / aspectRatio));
  int sampleCount = 100;
  int sampleDepth = 5;
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
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
  create_world<<<1, 1>>>(d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks((nx + tx - 1) / tx, (ny + ty - 1) / ty);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny, sampleCount, sampleDepth, d_camera,
                              d_world);
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
  free_world<<<1, 1>>>(d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));

  // useful for cuda-memcheck --leak-check full
  cudaDeviceReset();
}
