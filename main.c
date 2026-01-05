/*
 * Background Removal Tool
 * * Purpose: Isolates a graphic from its background using a boundary-initiated
 * flood fill algorithm with color tolerance and edge feathering.
 *
 * Usage:
 * ./remove_bg input.png output.png [tolerance 0-100] [feather_radius]
 */

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <mimalloc-override.h>

// --- Data Structures ---

typedef struct {
  int x, y;
} Point;

/**
 * Queue structure for BFS
 */
typedef struct {
  Point *items;
  int capacity;
  int front;
  int rear;
} Queue;

// --- Queue Implementation ---

Queue *createQueue(const int capacity) {
  auto const q = (Queue *)malloc(sizeof(Queue));
  q->items = (Point *)malloc(sizeof(Point) * (size_t)capacity);
  q->capacity = capacity;
  q->front = 0;
  q->rear = 0;
  return q;
}

bool isEmpty(Queue *q) {
  if (!q)
    return true;
  return q->front == q->rear;
}

void enqueue(Queue *q, const int x, const int y) {
  if (!q)
    return;
  if (q->rear == q->capacity) {
    printf("Error: Queue overflow. Increase capacity.\n");
    return;
  }
  q->items[q->rear++] = (Point){x, y};
}

Point dequeue(Queue *q) {
  if (isEmpty(q)) {
    printf("Error: Queue underflow.\n");
    return (Point){-1, -1};
  }
  return q->items[q->front++];
}

void freeQueue(Queue *q) {
  if (!q)
    return;

  if (q->items)
    free(q->items);
  free(q);
}

/*
 * color_distance_sq
 * Euclidean distance squared between two RGB pixels
 */
double color_distance_sq(const unsigned char *p1, const unsigned char *p2) {
  const double dr = (double)p1[0] - p2[0];
  const double dg = (double)p1[1] - p2[1];
  const double db = (double)p1[2] - p2[2];
  return dr * dr + dg * dg + db * db;
}

/*
 * refine_edges
 * Performs a box blur on the alpha channel to soften jagged edges
 * (antialiasing). Only affects pixels that are not fully transparent or fully
 * opaque to preserve details.
 */
void refine_edges(unsigned char *img, const int w, const int h,
                  const int channels) {
  auto const alpha_copy = (unsigned char *)malloc((unsigned long)(w * h));

  // Copy current alpha values
  for (auto i = 0; i < w * h; i++) {
    alpha_copy[i] = img[i * channels + 3];
  }

  constexpr auto kernel_size = 3; // 3x3 blur
  constexpr auto offset = kernel_size / 2;

  for (int y = offset; y < h - offset; y++) {
    for (int x = offset; x < w - offset; x++) {
      const int idx = (y * w + x) * channels + 3;

      // Skip fully opaque interior or fully transparent exterior to save cycles
      // Only blur the transition zone
      if (img[idx] == 0 || img[idx] == 255)
        continue;

      auto sum = 0;
      auto count = 0;

      for (int ky = -offset; ky <= offset; ky++) {
        for (int kx = -offset; kx <= offset; kx++) {
          const int neighbor_idx = (y + ky) * w + (x + kx);
          sum += alpha_copy[neighbor_idx];
          count++;
        }
      }
      img[idx] = (unsigned char)(sum / count);
    }
  }
  free(alpha_copy);
}

/*
 * anti_alias_edges
 * Scans image for object boundaries. If a boundary pixel is vastly different
 * in color from its neighbors (indicating noise or halo), it is smoothed.
 */
void anti_alias_edges(unsigned char *img, const int w, const int h,
                      const int channels) {
  // Create a copy to sample from so we don't read modified values
  auto const img_copy =
      (unsigned char *)malloc((unsigned long)(w * h * channels));
  if (!img_copy) {
    fprintf(stderr, "Memory allocation failed for AA pass.\n");
    return;
  }
  memcpy(img_copy, img, (unsigned long)(w * h * channels));

#pragma omp parallel for collapse(2)                                           \
    shared(img_copy, w, h, channels, img) default(private)
  for (auto y = 1; y < h - 1; y++) {
    for (auto x = 1; x < w - 1; x++) {
      const int idx = (y * w + x) * channels;

      // Only process pixels that are not fully transparent (part of the object)
      if (img_copy[idx + 3] == 0)
        continue;

      auto is_boundary = false;
      auto high_variance = false;

      auto r_sum = 0;
      auto g_sum = 0;
      auto b_sum = 0;
      auto count = 0;

      const int r = img_copy[idx];
      const int g = img_copy[idx + 1];
      const int b = img_copy[idx + 2];

      // 3x3 Neighborhood Check
      for (auto ky = -1; ky <= 1; ky++) {
        for (auto kx = -1; kx <= 1; kx++) {
          const int n_idx = ((y + ky) * w + (x + kx)) * channels;

          // Check if neighbor is transparent (implies we are on an edge)
          if (img_copy[n_idx + 3] == 0) {
            is_boundary = true;
          } else {
            constexpr auto diff_threshold_sq = 3600.0;
            // Check color difference
            const int nr = img_copy[n_idx];
            const int ng = img_copy[n_idx + 1];
            const int nb = img_copy[n_idx + 2];

            auto const dist_sq =
                (double)((r - nr) * (r - nr) + (g - ng) * (g - ng) +
                         (b - nb) * (b - nb));
            if (dist_sq > diff_threshold_sq) {
              high_variance = true;
            }

            // Accumulate for potential smoothing
            r_sum += nr;
            g_sum += ng;
            b_sum += nb;
            count++;
          }
        }
      }

      // Apply smoothing if it's a boundary pixel with high local color variance
      if (is_boundary && high_variance && count > 0) {
        constexpr auto a_sum = 0;
        img[idx] = (unsigned char)(r_sum / count);
        img[idx + 1] = (unsigned char)(g_sum / count);
        img[idx + 2] = (unsigned char)(b_sum / count);
        img[idx + 3] = (unsigned char)(a_sum / count);
      }
    }
  }
  free(img_copy);
}

/*
 * remove_background
 * Uses BFS flood fill from image borders to identify and remove background.
 * tolerance: percentage (0-100) of color difference allowed.
 */
void remove_background(unsigned char *img, const int w, const int h,
                       const int channels, const double tolerance) {
  if (channels < 3)
    return; // Must be RGB or RGBA

  // Convert percentage tolerance to distance squared threshold
  // Max distance is sqrt(255^2 * 3) approx 441.6
  constexpr auto max_dist = 441.67;
  const double threshold = tolerance / 100.0 * max_dist;
  const double threshold_sq = threshold * threshold;

  auto const visited = (bool *)calloc((unsigned long)(w * h), sizeof(bool));
  auto const q = createQueue(w * h);

  // Seed the queue with border pixels
  // We assume the top-left pixel (0,0) represents the background color
  const unsigned char *bg_ref = &img[0];

  // Add border pixels to queue if they match reference color
  for (auto y = 0; y < h; y++) {
    for (auto x = 0; x < w; x++) {
      if (x == 0 || x == w - 1 || y == 0 || y == h - 1) {
        const int idx = (y * w + x) * channels;
        if (color_distance_sq(&img[idx], bg_ref) <= threshold_sq) {
          enqueue(q, x, y);
          visited[y * w + x] = true;
        }
      }
    }
  }

  // Directions: Up, Down, Left, Right

  // BFS Flood Fill
  while (!isEmpty(q)) {
    const Point p = dequeue(q);
    const int current_idx = (p.y * w + p.x) * channels;

    // Set alpha to 0 (Transparent)
    // If image was RGB, we treat it as RGBA in memory (handled in main)
    img[current_idx + 3] = 0;

    // Check neighbors
    for (auto i = 0; i < 4; i++) {
      constexpr int dy[] = {-1, 1, 0, 0};
      constexpr int dx[] = {0, 0, -1, 1};
      const int nx = p.x + dx[i];
      const int ny = p.y + dy[i];

      if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
        const int n_idx = ny * w + nx;

        if (!visited[n_idx]) {
          const int pixel_offset = n_idx * channels;
          const double dist = color_distance_sq(&img[pixel_offset], bg_ref);

          if (dist <= threshold_sq) {
            visited[n_idx] = true;
            enqueue(q, nx, ny);
          } else {
            // Edge logic: If close to threshold, partial alpha?
            // For now, mark visited so we don't process again
            visited[n_idx] = true;
          }
        }
      }
    }
  }

  free(visited);
  freeQueue(q);
}

/*
 * remove_inner_background
 * Searches for "holes" inside the object (e.g. handle of a mug) that match
 * the background color but were not reached by the border flood fill.
 * Uses the same BFS strategy to ensure connected regions are removed cleanly.
 */
void remove_inner_background(unsigned char *img, int w, int h, int channels,
                             double tolerance) {
  if (channels < 3)
    return;

  constexpr auto max_dist = 441.67;
  const double threshold = tolerance / 100.0 * max_dist;
  const double threshold_sq = threshold * threshold;

  auto const visited = (bool *)calloc(w * h, sizeof(bool));
  Queue *q = createQueue(w * h); // Max capacity potentially needed

  // Reuse the Top-Left pixel as the reference color for internal holes
  // Note: Even if alpha was set to 0 in pass 1, RGB values are preserved.
  const unsigned char *bg_ref = &img[0];

  // Scan the entire image for seeds
  for (auto y = 0; y < h; y++) {
    for (auto x = 0; x < w; x++) {
      const int idx = y * w + x;
      const int pixel_offset = idx * channels;

      // Only process if:
      // 1. Not already visited (handled by this pass)
      // 2. Currently Opaque (not removed by pass 1)
      // 3. Matches background color
      if (!visited[idx] && img[pixel_offset + 3] != 0) {
        const double dist = color_distance_sq(&img[pixel_offset], bg_ref);

        if (dist <= threshold_sq) {
          // Found a seed (internal hole)
          enqueue(q, x, y);
          visited[idx] = true;

          // Execute BFS immediately for this component
          while (!isEmpty(q)) {
            const Point p = dequeue(q);
            const int current_idx = (p.y * w + p.x) * channels;

            // Remove (Set alpha to 0)
            img[current_idx + 3] = 0;

            for (auto i = 0; i < 4; i++) {
              constexpr int dy[] = {-1, 1, 0, 0};
              constexpr int dx[] = {0, 0, -1, 1};

              const int nx = p.x + dx[i];
              const int ny = p.y + dy[i];

              if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
                const int n_idx = ny * w + nx;

                if (!visited[n_idx]) {
                  const int n_offset = n_idx * channels;

                  // Must be opaque to consider removing
                  if (img[n_offset + 3] != 0) {
                    const double n_dist =
                        color_distance_sq(&img[n_offset], bg_ref);
                    if (n_dist <= threshold_sq) {
                      visited[n_idx] = true;
                      enqueue(q, nx, ny);
                    }
                  } else {
                    // Already transparent (from pass 1), just mark visited
                    visited[n_idx] = true;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  free(visited);
  freeQueue(q);
}

int main(const int argc, char **argv) {
  if (argc < 3) {
    printf("Usage: %s <input> <output> [tolerance %%] [smooth]\n", argv[0]);
    printf("Default tolerance: 15.0\n");
    return 1;
  }

  const char *input_path = argv[1];
  const char *output_path = argv[2];
  const double tolerance = argc > 3 ? (double)strtof(argv[3], nullptr) : 15.0;
  const int smooth_pass = argc > 4 ? strtol(argv[4], nullptr, 10) : 1;

  int w, h, c;
  // Force load as 4 channels (RGBA)
  unsigned char *img = stbi_load(input_path, &w, &h, &c, 4);

  if (!img) {
    printf("Error loading image: %s\n", input_path);
    return 1;
  }

  fprintf(stderr, "Image loaded: %s (%dx%d), Tolerance: %.2f%%\n", input_path,
          w, h, tolerance);
  remove_background(img, w, h, 4, tolerance);

  // Clean up inner background
  printf("Removing inner background...\n");
  remove_inner_background(img, w, h, 4, tolerance);

  if (smooth_pass) {
    printf("Refining edges...\n");
    refine_edges(img, w, h, 4);
    anti_alias_edges(img, w, h, 4);
  }

  if (stbi_write_png(output_path, w, h, 4, img, w * 4)) {
    printf("Success: Saved to %s\n", output_path);
  } else {
    printf("Error writing output file.\n");
  }

  stbi_image_free(img);
  return 0;
}
