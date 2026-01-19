args <- commandArgs(trailingOnly = TRUE)
size <- if (length(args) >= 1) as.integer(args[1]) else 1024L
iters <- if (length(args) >= 2) as.integer(args[2]) else 5L
warmup <- if (length(args) >= 3) as.integer(args[3]) else 1L

set.seed(1)
a <- matrix(runif(size * size), nrow = size, ncol = size)
b <- matrix(runif(size * size), nrow = size, ncol = size)

if (warmup > 0) {
  for (i in seq_len(warmup)) {
    c <- a %*% b
  }
}

times <- numeric(iters)
if (iters > 0) {
  for (i in seq_len(iters)) {
    t <- system.time({
      c <- a %*% b
    })
    times[i] <- as.numeric(t["elapsed"])
  }
}

median_s <- if (iters > 0) median(times) else NA_real_
gflops <- (2.0 * size * size * size) / (median_s * 1e9)

cat(sprintf("tool=r op=matmul shape=%dx%dx%d metric=latency_s value=%.6f\n",
            size, size, size, median_s))
cat(sprintf("tool=r op=matmul shape=%dx%dx%d metric=gflops value=%.3f\n",
            size, size, size, gflops))
