# Reproduce R-based figures from the Source Data workbook.
# Put this script and `Source Data.xlsx` in the same directory, then run:
#   Rscript plot_figures_from_source.R
# Required packages: readxl, ggplot2, dplyr, tidyr, patchwork, ggdendro, ggnewscale, lme4

suppressPackageStartupMessages({
  library(readxl); library(ggplot2); library(dplyr); library(tidyr); library(patchwork);
  library(ggdendro); library(ggnewscale); library(lme4)
})

# Suppress empty .pdf with null graphic device
pdf(NULL)


source_file <- "Source Data.xlsx"
out_dir <- "figures"
dir.create(out_dir, showWarnings = FALSE)
if (!file.exists(source_file)) stop(paste("Could not find", source_file, "in the current directory."))

primary_cluster_cols <- c("1"="#440154FF", "2"="#22A884FF", "3"="#D55E00")
supp_cluster_cols3 <- c("1"="#6A1B6A", "2"="#22B89A", "3"="#E67E00")
cluster_cols6 <- c(supp_cluster_cols3,"4"="#4C78A8","5"="#D4A72C","6"="#CC6677")

###############################################################
# Fig. 2
###############################################################

fig2 <- read_excel(source_file, sheet = "Fig. 2")

# Adapt source-data column names to the variables used
fig2 <- fig2 %>%
  transmute(
    ID = ID,
    Cluster = as.numeric(cluster),
    Block = as.numeric(block),
    shift = shift
  )

K_FINAL <- 3

# Compute cluster sizes
cluster_sizes <- fig2 %>%
  distinct(ID, Cluster) %>%
  dplyr::count(Cluster, name = "n")

cluster_sizes$Label <- c(
  "alternating reward-driven",
  "progressive cue-driven",
  "unstable"
)

# Create named vector for labeller
labeller_vec <- setNames(
  paste0(
    "Cluster ", cluster_sizes$Cluster,
    ":\n ", cluster_sizes$Label,
    "\n(n = ", cluster_sizes$n, ")"
  ),
  cluster_sizes$Cluster
)

# Get y-limits from the data
y_max <- max(
  fig2$shift,
  na.rm = TRUE
)

y_min <- min(
  fig2$shift,
  na.rm = TRUE
)

# Get the background coding
background_df <- data.frame(
  Cluster = c(1, 2, 3)
)

#Plot
p_trajectories_large_neutral <- ggplot(
  fig2,
  aes(
    x = as.numeric(Block),
    y = shift,
    group = ID,
    color = factor(Cluster)
  )
) +
  
  # background bands
  geom_rect(
    aes(
      xmin = -Inf,
      xmax = Inf,
      ymin = 0,
      ymax = Inf
    ),
    inherit.aes = TRUE,
    fill = "lightblue",
    color = 'transparent',
    alpha = 0.035
  ) +
  
  geom_rect(
    aes(
      xmin = -Inf,
      xmax = Inf,
      ymin = -Inf,
      ymax = 0
    ),
    inherit.aes = TRUE,
    fill = "mistyrose",
    color = 'transparent',
    alpha = 0.035
  ) +
  
  # zero line
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    color = "black",
    linewidth = 0.7
  ) +
  
  # individual trajectories
  geom_line(
    alpha = 0.3,
    linewidth = 0.6
  ) +
  
  # mean ± SE ribbon
  stat_summary(
    aes(
      group = Cluster,
      fill = factor(Cluster)
    ),
    fun.data = mean_se,
    geom = "ribbon",
    alpha = 0.20,
    color = NA
  ) +
  
  # mean trajectory
  stat_summary(
    aes(group = Cluster),
    fun = mean,
    geom = "line",
    linewidth = 2.2
  ) +
  
  facet_wrap(
    ~Cluster,
    labeller = as_labeller(labeller_vec)
  ) +
  
  scale_color_manual(
    values = c(
      "#440154FF",
      "#22A884FF",
      "#D55E00"
    ),
    name = "Cluster"
  ) +
  
  scale_fill_manual(
    values = c(
      "#440154FF",
      "#22A884FF",
      "#D55E00"
    ),
    guide = "none"
  ) +
  
  labs(
    title = "Consensus Clustering Solution applied to Large > Neutral Reward Contrast",
    subtitle = paste(
      "K =", K_FINAL,
      "| Thick line = mean ± SE"
    ),
    x = "Learning Block",
    y = "Reward-to-cue Shift \nNAcc BOLD beta weights"
  ) +
  
  annotate(
    "text",
    x = 4.15,
    y = y_max * 0.92,
    label = "Cue-driven value representations",
    hjust = 1,
    size = 5,
    fontface = "bold"
  ) +
  
  annotate(
    "text",
    x = 4.15,
    y = y_min * 0.92,
    label = "Reward-driven value representations",
    hjust = 1,
    size = 5,
    fontface = "bold"
  ) +
  
  theme_minimal(base_size = 16) +
  
  theme(
    plot.title = element_text(
      size = 20,
      face = "bold"
    ),
    plot.subtitle = element_text(size = 16),
    strip.text = element_text(
      size = 16,
      face = "bold"
    ),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    legend.position = "bottom",
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 13)
  )


ggsave(
  file.path(out_dir, "Fig2.pdf"),
  p_trajectories_large_neutral,
  width = 17,
  height = 8
)

ggsave(
  file.path(out_dir, "Fig2.png"),
  p_trajectories_large_neutral,
  width = 17,
  height = 8,
  dpi = 300
)

ggsave(
  file.path(out_dir, "Fig2.svg"),
  p_trajectories_large_neutral,
  width = 17,
  height = 8
)

# ============================================================
# Supplementary Figure 8 
# ============================================================

# ------------------------------------------------------------
# Read source data
# ------------------------------------------------------------

bootstrap <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 8a"
)

silhouette <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 8b"
)

pca <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 8c"
)


# ------------------------------------------------------------
# Common theme
# ------------------------------------------------------------

common_theme <- theme_classic(base_size = 11) +
  theme(
    plot.title = element_text(
      size = 13,
      face = "plain",
      hjust = 0
    ),
    plot.tag = element_text(
      size = 12,
      face = "bold"
    ),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    plot.margin = margin(8, 8, 8, 8)
  )


# ============================================================
# Panel a — Bootstrap stability
# ============================================================

median_ari <- median(
  bootstrap$ARI,
  na.rm = TRUE
)

p_a <- ggplot(
  bootstrap,
  aes(x = ARI)
) +
  
  geom_histogram(
    bins = 30,
    fill = "#5B83BD",
    color = "white",
    linewidth = 0.3
  ) +
  
  geom_vline(
    xintercept = median_ari,
    linetype = "dashed",
    linewidth = 0.8,
    color = "black"
  ) +
  
  labs(
    title = "Bootstrap stability of clustering solution",
    x = "Adjusted Rand Index (ARI)",
    y = "Bootstrap iterations",
    tag = "a"
  ) +
  
  scale_x_continuous(
    breaks = c(0.25, 0.50, 0.75, 1.00)
  ) +
  
  common_theme


# ============================================================
# Panel b — Silhouette analysis
# ============================================================

silhouette <- silhouette %>%
  mutate(
    cluster = factor(
      cluster,
      levels = c(1, 2, 3)
    )
  )


# Arrange participants by cluster and silhouette width
silhouette_plot <- silhouette %>%
  arrange(
    cluster,
    desc(silhouette_width)
  ) %>%
  mutate(
    plot_order = row_number()
  )


# Mean silhouette width
mean_silhouette <- mean(
  silhouette$silhouette_width,
  na.rm = TRUE
)


# Actual observed range:
# -0.09639873 to 0.64797465
#
# Keep a little space below the minimum value.
silhouette_ymin <- min(
  -0.15,
  min(
    silhouette$silhouette_width,
    na.rm = TRUE
  ) - 0.03
)


p_b <- ggplot(
  silhouette_plot,
  aes(
    x = plot_order,
    y = silhouette_width,
    fill = cluster
  )
) +
  
  geom_col(
    width = 0.82
  ) +
  
  # Average silhouette width
  geom_hline(
    yintercept = mean_silhouette,
    linetype = "dashed",
    linewidth = 0.7,
    color = "#FF5A5A"
  ) +
  
  scale_fill_manual(
    values = supp_cluster_cols3,
    name = "Cluster"
  ) +
  
  # Leave a small amount of room around the first/last bars
  scale_x_continuous(
    expand = expansion(
      add = c(0.5, 0.5)
    )
  ) +
  
  # IMPORTANT:
  # Do not use limits here because scale limits can remove
  # observations before geom_col() is drawn.
  scale_y_continuous(
    breaks = seq(
      -0.25,
      1,
      0.25
    )
  ) +
  
  # Zoom without removing observations
  coord_cartesian(
    ylim = c(
      silhouette_ymin,
      1.05
    )
  ) +
  
  labs(
    title = paste0(
      "Silhouette Plot for K = 3\n",
      "Average Silhouette Width = ",
      sprintf(
        "%.2f",
        mean_silhouette
      )
    ),
    x = NULL,
    y = "Silhouette width Si",
    tag = "b"
  ) +
  
  theme_gray(base_size = 11) +
  
  theme(
    plot.title = element_text(
      size = 13,
      face = "plain",
      hjust = 0
    ),
    plot.tag = element_text(
      size = 12,
      face = "bold"
    ),
    axis.title.y = element_text(
      size = 10
    ),
    axis.text.y = element_text(
      size = 9
    ),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    legend.position = "right",
    legend.title = element_text(
      size = 10
    ),
    legend.text = element_text(
      size = 9
    ),
    plot.margin = margin(
      8, 8, 8, 8
    )
  )


# ============================================================
# Panel c — PCA
# ============================================================

pca <- pca %>%
  mutate(
    cluster = factor(
      cluster,
      levels = c(1, 2, 3)
    )
  )


p_c <- ggplot(
  pca,
  aes(
    x = PC1,
    y = PC2,
    color = cluster
  )
) +
  
  stat_ellipse(
    aes(group = cluster),
    level = 0.68,
    linewidth = 0.8,
    show.legend = FALSE
  ) +
  
  geom_point(
    size = 2.2,
    alpha = 0.80
  ) +
  
  scale_color_manual(
    values = supp_cluster_cols3
  ) +
  
  labs(
    title = "Cluster Separation in PCA Space",
    x = "PC1 (33.9% variance)",
    y = "PC2 (27.7% variance)",
    tag = "c"
  ) +
  
  theme_minimal(base_size = 11) +
  
  theme(
    plot.title = element_text(
      size = 13,
      face = "plain",
      hjust = 0
    ),
    plot.tag = element_text(
      size = 12,
      face = "bold"
    ),
    axis.title = element_text(
      size = 10
    ),
    axis.text = element_text(
      size = 8
    ),
    panel.grid.major = element_line(
      color = "grey90",
      linewidth = 0.4
    ),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(
      8, 8, 8, 8
    )
  )


# ============================================================
# Combine panels
# ============================================================

FigS8 <- p_a + p_b + p_c


# ============================================================
# Save
# ============================================================

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig8.pdf"
  ),
  FigS8,
  width = 16,
  height = 4,
  units = "in"
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig8.png"
  ),
  FigS8,
  width = 16,
  height = 4,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig8.svg"
  ),
  FigS8,
  width = 16,
  height = 4,
  units = "in",
  dpi = 600
)

# ============================================================
# Supplementary Figure 7 
# ============================================================

panel_margin <- margin(t = 8, r = 8, b = 8, l = 8)

panel_tag_theme <- theme(
  plot.tag = element_text(face = "bold", size = 12),
  plot.margin = panel_margin
)

# ============================================================
# PANELS a-e: CONSENSUS MATRICES
# ============================================================

read_consensus <- function(panel) {
  dat <- read_excel(source_file, sheet = paste0("Supplementary Fig. 7", panel))
  IDs <- as.character(dat[[1]])
  clusters <- as.integer(dat[[2]])
  M <- as.matrix(dat[, 3:ncol(dat)])
  storage.mode(M) <- "numeric"
  rownames(M) <- IDs
  colnames(M) <- IDs
  list(matrix = M, ID = IDs, cluster = clusters)
}

plot_consensus <- function(panel, k, panel_letter) {
  dat <- read_consensus(panel)
  M <- dat$matrix
  clusters <- dat$cluster
  IDs <- dat$ID
  n <- nrow(M)
  
  matrix_df <- expand.grid(x = seq_len(n), y = seq_len(n))
  matrix_df$consensus <- as.vector(M)
  
  strip_df <- data.frame(
    x = seq_len(n),
    cluster = factor(clusters, levels = 1:6)
  )
  
  hc <- hclust(as.dist(1 - M), method = "average")
  dendro <- ggdendro::dendro_data(hc, type = "rectangle")
  dend_segments <- ggdendro::segment(dendro)
  
  matrix_top <- n + 0.5
  strip_bottom <- n + 0.8
  strip_top <- n + 2.2
  dend_bottom <- strip_top
  dend_top <- n + 17
  
  dend_y_min <- min(c(dend_segments$y, dend_segments$yend), na.rm = TRUE)
  dend_y_max <- max(c(dend_segments$y, dend_segments$yend), na.rm = TRUE)
  
  rescale_dend_y <- function(z) {
    dend_bottom +
      (z - dend_y_min) / (dend_y_max - dend_y_min) *
      (dend_top - dend_bottom)
  }
  
  dend_segments <- dend_segments %>%
    mutate(
      y_plot = rescale_dend_y(y),
      yend_plot = rescale_dend_y(yend)
    )
  
  p <- ggplot() +
    geom_tile(
      data = matrix_df,
      aes(x = x, y = y, fill = consensus),
      width = 1,
      height = 1
    ) +
    scale_fill_gradient(
      low = "azure",
      high = "blue",
      limits = c(0, 1),
      name = NULL
    ) +
    ggnewscale::new_scale_fill() +
    geom_rect(
      data = strip_df,
      aes(
        xmin = x - 0.5,
        xmax = x + 0.5,
        ymin = strip_bottom,
        ymax = strip_top,
        fill = cluster
      ),
      color = NA
    ) +
    scale_fill_manual(
      values = cluster_cols6,
      drop = FALSE
    ) +
    geom_segment(
      data = dend_segments,
      aes(
        x = x,
        y = y_plot,
        xend = xend,
        yend = yend_plot
      ),
      inherit.aes = FALSE,
      linewidth = 0.3,
      color = "grey40"
    ) +
    scale_x_continuous(
      limits = c(0.5, n + 0.5),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(0.5, dend_top),
      expand = c(0, 0)
    ) +
    coord_fixed(ratio = 1, clip = "off") +
    labs(
      title = paste0("consensus matrix k=", k),
      tag = panel_letter
    ) +
    theme_void() +
    theme(
      legend.position = "none",
      plot.title = element_text(
        size = 9,
        face = "bold",
        hjust = 0.5,
        margin = margin(b = 4)
      ),
      plot.tag = element_text(
        size = 12,
        face = "bold"
      ),
      plot.margin = panel_margin
    )
  
  return(p)
}

p_a <- plot_consensus(panel = "a", k = 2, panel_letter = "a")
p_b <- plot_consensus(panel = "b", k = 3, panel_letter = "b")
p_c <- plot_consensus(panel = "c", k = 4, panel_letter = "c")
p_d <- plot_consensus(panel = "d", k = 5, panel_letter = "d")
p_e <- plot_consensus(panel = "e", k = 6, panel_letter = "e")

# ============================================================
# PANEL f: CONSENSUS CDF
# ============================================================

cdf <- read_excel(source_file, sheet = "Supplementary Fig. 7f")

cdf_long <- cdf %>%
  pivot_longer(
    cols = k2:k6,
    names_to = "k",
    values_to = "CDF"
  ) %>%
  mutate(
    k = factor(
      k,
      levels = paste0("k", 2:6),
      labels = 2:6
    )
  )

p_f <- ggplot(
  cdf_long,
  aes(x = index, y = CDF, color = k)
) +
  geom_line(linewidth = 0.65) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.2),
    expand = c(0, 0)
  ) +
  labs(
    title = "consensus CDF",
    x = "consensus index",
    y = "CDF",
    color = NULL,
    tag = "f"
  ) +
  theme_classic(base_size = 9) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.tag = element_text(face = "bold", size = 12),
    legend.position = c(0.82, 0.34),
    legend.background = element_rect(
      fill = "white",
      color = "grey60",
      linewidth = 0.3
    ),
    plot.margin = panel_margin
  )

# ============================================================
# PANEL g: DELTA AREA
# ============================================================

delta <- read_excel(source_file, sheet = "Supplementary Fig. 7g")

p_g <- ggplot(
  delta,
  aes(x = k, y = relative_change)
) +
  geom_line(
    color = "grey70",
    linewidth = 0.7
  ) +
  geom_point(
    shape = 21,
    size = 2,
    fill = "white",
    color = "grey50",
    stroke = 0.6
  ) +
  scale_x_continuous(breaks = 2:6) +
  labs(
    title = "Delta area",
    x = "k",
    y = "relative change in area under CDF curve",
    tag = "g"
  ) +
  theme_classic(base_size = 9) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.tag = element_text(face = "bold", size = 12),
    plot.margin = panel_margin
  )

# ============================================================
# PANEL h: TRACKING PLOT
# ============================================================

tracking <- read_excel(source_file, sheet = "Supplementary Fig. 7h")

tracking_long <- tracking %>%
  pivot_longer(
    cols = k2:k6,
    names_to = "k",
    values_to = "cluster"
  ) %>%
  mutate(
    k = as.numeric(sub("k", "", k)),
    cluster = factor(cluster),
    order = factor(order, levels = tracking$order)
  )

p_h <- ggplot(
  tracking_long,
  aes(
    x = order,
    y = factor(k),
    fill = cluster
  )
) +
  geom_tile(
    width = 1,
    height = 1
  ) +
  scale_fill_manual(values = cluster_cols6) +
  scale_y_discrete(
    limits = rev,
    expand = c(0, 0)
  ) +
  scale_x_discrete(
    expand = c(0, 0)
  ) +
  labs(
    title = "tracking plot",
    x = "samples",
    y = "k",
    tag = "h"
  ) +
  theme_classic(base_size = 9) +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    plot.tag = element_text(face = "bold", size = 12),
    axis.text.x = element_blank(),
    axis.ticks.x = element_line(),
    legend.position = "none",
    plot.margin = panel_margin
  )

# ============================================================
# PANEL i: ASSIGNED vs NEAREST COMPETING CLUSTER
# ============================================================

competing <- read_excel(source_file, sheet = "Supplementary Fig. 7i")

p_i <- ggplot(
  competing,
  aes(
    x = consensus_assigned,
    y = consensus_competing,
    color = factor(cluster)
  )
) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed",
    color = "black",
    linewidth = 0.6
  ) +
  geom_point(
    size = 2.4,
    alpha = 0.85
  ) +
  scale_color_manual(
    values = cluster_cols6[c("1", "2", "3")],
    name = "Cluster"
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = c(0, 0)
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = c(0, 0)
  ) +
  coord_fixed(ratio = 1) +
  labs(
    x = "Consensus with assigned cluster",
    y = "Consensus with nearest competing cluster",
    tag = "i"
  ) +
  theme_classic(base_size = 9) +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 12
    ),
    legend.position = c(0.10, 0.90),
    legend.background = element_rect(
      fill = "white",
      color = NA
    ),
    plot.margin = panel_margin
  )

# ============================================================
# FINAL 3 x 3 LAYOUT
# ============================================================

figure_design <- "
ABC
DEF
GHI
"

FigS7 <-
  p_a + p_b + p_c +
  p_d + p_e + p_f +
  p_g + p_h + p_i +
  plot_layout(
    design = figure_design,
    widths = c(1, 1, 1),
    heights = c(1, 1, 1)
  )


ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig7.svg"
  ),
  FigS7,
  width = 12,
  height = 12,
  units = "in",
  limitsize = FALSE
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig7.pdf"
  ),
  FigS7,
  width = 12,
  height = 12,
  units = "in",
  limitsize = FALSE
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig7.png"
  ),
  FigS7,
  width = 12,
  height = 12,
  units = "in",
  limitsize = FALSE
)

# ============================================================
# SUPPLEMENTARY FIGURE 13
# ============================================================

s13 <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 13"
) %>%
  mutate(
    cluster = as.numeric(cluster),
    block = as.numeric(block)
  )

# ------------------------------------------------------------
# Helper function
# Uses EXACTLY the same plotting structure/style as Figure 2
# ------------------------------------------------------------

plot_alternative_contrast <- function(data, contrast_name, plot_title) {
  
  # Select contrast
  df_plot <- data %>%
    filter(contrast == contrast_name) %>%
    transmute(
      ID = ID,
      Group = cluster,
      Block = block,
      BOLD_SHIFT = reward_to_cue_shift
    )
  
  # Cluster sizes
  cluster_sizes_alt <- df_plot %>%
    distinct(ID, Group) %>%
    dplyr::count(Group, name = "n") %>%
    arrange(Group)
  
  cluster_sizes_alt$Label <- c(
    "alternating reward-driven",
    "progressive cue-driven",
    "unstable"
  )
  
  # Facet labels
  labeller_vec_alt <- setNames(
    paste0(
      "Cluster ", cluster_sizes_alt$Group,
      ":\n ", cluster_sizes_alt$Label,
      "\n(n = ", cluster_sizes_alt$n, ")"
    ),
    cluster_sizes_alt$Group
  )
  
  # Get y limits independently for each contrast
  y_max_alt <- max(
    df_plot$BOLD_SHIFT,
    na.rm = TRUE
  )
  
  y_min_alt <- min(
    df_plot$BOLD_SHIFT,
    na.rm = TRUE
  )
  
  # Plot
  ggplot(
    df_plot,
    aes(
      x = as.numeric(Block),
      y = BOLD_SHIFT,
      group = ID,
      color = factor(Group)
    )
  ) +
    
    # background bands
    geom_rect(
      aes(
        xmin = -Inf,
        xmax = Inf,
        ymin = 0,
        ymax = Inf
      ),
      inherit.aes = TRUE,
      fill = "lightblue",
      color = 'transparent',
      alpha = 0.035
    ) +
    
    geom_rect(
      aes(
        xmin = -Inf,
        xmax = Inf,
        ymin = -Inf,
        ymax = 0
      ),
      inherit.aes = TRUE,
      fill = "mistyrose",
      color = 'transparent',
      alpha = 0.035
    ) +
    
    # Zero line
    geom_hline(
      yintercept = 0,
      linetype = "dashed",
      color = "black",
      linewidth = 0.7
    ) +
    
    # Individual trajectories
    geom_line(
      alpha = 0.3,
      linewidth = 0.6
    ) +
    
    # Mean +/- SE ribbon
    stat_summary(
      aes(
        group = Group,
        fill = factor(Group)
      ),
      fun.data = mean_se,
      geom = "ribbon",
      alpha = 0.20,
      color = NA
    ) +
    
    # Mean trajectory
    stat_summary(
      aes(group = Group),
      fun = mean,
      geom = "line",
      linewidth = 2.2
    ) +
    
    # Same cluster facets as Figure 2
    facet_wrap(
      ~Group,
      labeller = as_labeller(labeller_vec_alt)
    ) +
    
    # Same colors as Figure 2
    scale_color_manual(
      values = c(
        "#440154FF",
        "#22A884FF",
        "#D55E00"
      ),
      name = "Cluster"
    ) +
    
    scale_fill_manual(
      values = c(
        "#440154FF",
        "#22A884FF",
        "#D55E00"
      ),
      guide = "none"
    ) +
    
    scale_x_continuous(
      breaks = 1:4
    ) +
    
    labs(
      title = plot_title,
      subtitle = paste(
        "K =", K_FINAL,
        "| Thick line = mean +/- SE"
      ),
      x = "Learning Block",
      y = "Reward-to-cue Shift \nNAcc BOLD beta weights"
    ) +
    
    # Same interpretation labels as Figure 2
    annotate(
      "text",
      x = 4.15,
      y = y_max_alt * 0.92,
      label = "Cue-driven value representations",
      hjust = 1,
      size = 5,
      fontface = "bold"
    ) +
    
    annotate(
      "text",
      x = 4.15,
      y = y_min_alt * 0.92,
      label = "Reward-driven value representations",
      hjust = 1,
      size = 5,
      fontface = "bold"
    ) +
    
    # Same theme as Figure 2
    theme_minimal(base_size = 16) +
    
    theme(
      plot.title = element_text(
        size = 20,
        face = "bold"
      ),
      plot.subtitle = element_text(
        size = 16
      ),
      strip.text = element_text(
        size = 16,
        face = "bold"
      ),
      axis.title = element_text(
        size = 16
      ),
      axis.text = element_text(
        size = 14
      ),
      legend.position = "bottom",
      legend.title = element_text(
        size = 14
      ),
      legend.text = element_text(
        size = 13
      )
    )
}


# ============================================================
# CREATE THE THREE CONTRAST PANELS
# ============================================================

p13_small <- plot_alternative_contrast(
  data = s13,
  contrast_name = "small_minus_neutral",
  plot_title = "Consensus Clustering Solution applied to Small > Neutral Reward Contrast"
)

p13_medium <- plot_alternative_contrast(
  data = s13,
  contrast_name = "medium_minus_neutral",
  plot_title = "Consensus Clustering Solution applied to Medium > Neutral Reward Contrast"
)

p13_large_small <- plot_alternative_contrast(
  data = s13,
  contrast_name = "large_minus_small",
  plot_title = "Consensus Clustering Solution applied to Large > Small Reward Contrast"
)


# ============================================================
# PATCHWORK
# Three rows, one contrast per row
# ============================================================

# ============================================================
# PATCHWORK
# Three rows, one contrast per row
# ============================================================

FigS13 <-
  p13_medium /
  p13_small /
  p13_large_small +
  plot_layout(
    ncol = 1,
    guides = "collect"
  ) +
  plot_annotation(
    tag_levels = "a"
  ) &
  theme(
    legend.position = "bottom",
    plot.tag = element_text(
      face = "bold",
      size = 22
    )
  )



# ============================================================
# SAVE
# ============================================================

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig13.pdf"
  ),
  FigS13,
  width = 17,
  height = 24,
  units = "in"
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig13.png"
  ),
  FigS13,
  width = 17,
  height = 24,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig13.svg"
  ),
  FigS13,
  width = 17,
  height = 24,
  units = "in"
)



# ============================================================
# SUPPLEMENTARY FIGURE 5
# Cue phase, outcome phase, and reward-to-cue shift
# ============================================================

figS5_data <- read_excel(source_file, sheet = "Supplementary Fig. 5") %>%
  transmute(
    ID = ID,
    Block = as.numeric(block),
    cue = cue_contrast,
    outcome = outcome_contrast,
    shift = shift
  )

plot_individual_phase <- function(data, yvar, panel_title, ylab) {
  
  df_plot <- data %>%
    transmute(
      ID = ID,
      Block = Block,
      value = .data[[yvar]]
    )
  
  ggplot(
    df_plot,
    aes(
      x = as.numeric(Block),
      y = value,
      group = ID
    )
  ) +
    
    # Individual trajectories
    geom_line(
      alpha = 0.15,
      linewidth = 0.6,
      color = "grey50"
    ) +
    
    # Median trajectory
    stat_summary(
      aes(group = 1),
      fun = median,
      geom = "line",
      linewidth = 1.5,
      color = "darkblue"
    ) +
    
    scale_x_continuous(
      breaks = 1:4
    ) +
    
    labs(
      title = panel_title,
      x = "Block",
      y = ylab
    ) +
    
    coord_cartesian(
      ylim = c(-4.5, 3)
    ) +
    
    theme_minimal(base_size = 13) +
    
    theme(
      plot.title = element_text(
        hjust = 0.5,
        size = 13
      ),
      plot.tag = element_text(
        face = "bold",
        size = 12
      )
    )
}


p_S5a <- plot_individual_phase(
  figS5_data,
  "cue",
  "Cue phase",
  "NAcc BOLD Beta Weights"
)

p_S5b <- plot_individual_phase(
  figS5_data,
  "outcome",
  "Outcome phase",
  "NAcc BOLD Beta Weights"
)

p_S5c <- plot_individual_phase(
  figS5_data,
  "shift",
  "Reward-to-cue shift",
  "Reward-to-cue shift\nNAcc BOLD Beta Weights"
)


FigS5 <-
  p_S5a + p_S5b + p_S5c +
  plot_annotation(
    tag_levels = "a"
  )



ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig5.pdf"
  ),
  FigS5,
  width = 14,
  height = 4.2,
  units = "in"
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig5.png"
  ),
  FigS5,
  width = 14,
  height = 4.2,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig5.svg"
  ),
  FigS5,
  width = 14,
  height = 4.2,
  units = "in"
)


# ============================================================
# SUPPLEMENTARY FIGURE 6
# ============================================================

# Subject-specific random effects
beta0 <- -8.313804e-16
beta1 <- 3.411417e-16

df_slopes <- read_excel(source_file, sheet = "Supplementary Fig. 6") 

# ---------------------------------------------------------
# Panel b: Distribution of subject-specific slopes
# ---------------------------------------------------------

p_slopes <- ggplot(
  df_slopes,
  aes(x = subject_slope)
) +
  
  geom_density(
    fill = "grey80",
    color = "grey30",
    linewidth = 0.6
  ) +
  
  # Zero slope
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.6
  ) +
  
  # Population-level slope
  geom_vline(
    xintercept = beta1,
    linewidth = 0.9
  ) +
  
  labs(
    x = "Subject-specific slope",
    y = "Density"
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.line = element_line(linewidth = 0.5),
    axis.ticks = element_line(linewidth = 0.5),
    plot.margin = margin(5, 5, 5, 5)
  )


# Create subject-specific fitted trajectories
df_fitted <- expand.grid(
  ID = df_slopes$ID,
  Block = 1:4
) %>%
  left_join(
    df_slopes,
    by = "ID"
  ) %>%
  mutate(
    fitted = subject_intercept + subject_slope * Block,
    fitted_change = fitted -
      (subject_intercept + subject_slope * 1)
  )


# ---------------------------------------------------------
# Population-level trajectory centered at Block 1
# ---------------------------------------------------------

df_fixed <- data.frame(
  Block = 1:4
) %>%
  mutate(
    fitted_change = beta1 * (Block - 1)
  )

# ---------------------------------------------------------
# Panel c: Centered subject-specific fitted trajectories
# ---------------------------------------------------------

p_trajectories <- ggplot(
  df_fitted,
  aes(
    x = Block,
    y = fitted_change,
    group = ID
  )
) +
  
  # Subject-specific fitted trajectories
  geom_line(
    color = "#377EB8",
    alpha = 0.45,
    linewidth = 0.7
  ) +
  
  # Population-level fixed-effect trajectory
  geom_line(
    data = df_fixed,
    aes(
      x = Block,
      y = fitted_change,
      group = 1
    ),
    color = "#222222",
    linewidth = 1.2
  ) +
  
  # Zero-change reference
  geom_hline(
    yintercept = 0,
    linetype = "dashed",
    linewidth = 0.5,
    color = "grey50"
  ) +
  
  scale_x_continuous(
    breaks = 1:4
  ) +
  
  labs(
    x = "MID block",
    y = "Fitted change from Block 1"
  ) +
  
  theme_classic(base_size = 12) +
  
  theme(
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.line = element_line(linewidth = 0.5),
    axis.ticks = element_line(linewidth = 0.5),
    plot.margin = margin(5, 5, 5, 5)
  )


# ---------------------------------------------------------
# Combine panels b and c only
# ---------------------------------------------------------

# Add the desired panel labels explicitly.
# patchwork::plot_annotation(tag_levels = ...) expects a tag sequence
# type such as "a" or "A"; it cannot start an automatic sequence at "b".
p_slopes <- p_slopes +
  labs(tag = "b") +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 12
    )
  )

p_trajectories <- p_trajectories +
  labs(tag = "c") +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 12
    )
  )

FigS6 <- p_slopes + p_trajectories



ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig6.pdf"
  ),
  FigS6,
  width = 8,
  height = 3.5,
  units = "in"
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig6.png"
  ),
  FigS6,
  width = 8,
  height = 3.5,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(
    out_dir,
    "Supplementary_Fig6.svg"
  ),
  FigS6,
  width = 8,
  height = 3.5,
  units = "in"
)




# ============================================================
# SUPPLEMENTARY FIGURE 9
# Median reaction time by cluster, block, and cue magnitude
# ============================================================

s9 <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 9"
) %>%
  mutate(
    cluster = factor(cluster, levels = c(1, 2, 3)),
    block = as.numeric(block),
    cue = factor(
      cue,
      levels = c(
        "neutral",
        "small_reward",
        "medium_reward",
        "large_reward"
      ),
      labels = c(
        "Neutral",
        "Small reward",
        "Medium reward",
        "Large reward"
      )
    ),
    RT_ms = RT_med * 1000
  )

cluster_labels_s9s10 <- c(
  "1" = "Cluster 1 · reward-driven (n=27)",
  "2" = "Cluster 2 · cue-driven (n=17)",
  "3" = "Cluster 3 · unstable (n=13)"
)

# Mean +/- SE across participants for each cluster x block x cue
s9_summary <- s9 %>%
  group_by(cluster, block, cue) %>%
  summarise(
    mean_RT = mean(RT_ms, na.rm = TRUE),
    se_RT = sd(RT_ms, na.rm = TRUE) / sqrt(sum(!is.na(RT_ms))),
    .groups = "drop"
  )

FigS9 <- ggplot(
  s9_summary,
  aes(
    x = block,
    y = mean_RT,
    color = cluster,
    group = cluster
  )
) +
  geom_errorbar(
    aes(
      ymin = mean_RT - se_RT,
      ymax = mean_RT + se_RT
    ),
    width = 0.08,
    linewidth = 0.45
  ) +
  geom_line(
    linewidth = 0.8
  ) +
  geom_point(
    size = 2.0
  ) +
  facet_wrap(
    ~cue,
    nrow = 1
  ) +
  scale_color_manual(
    values = primary_cluster_cols,
    labels = cluster_labels_s9s10,
    name = NULL
  ) +
  scale_x_continuous(
    breaks = 1:4
  ) +
  scale_y_continuous(
    breaks = seq(300, 420, 20)
  ) +
  coord_cartesian(
    ylim = c(300, 430)
  ) +
  labs(
    x = "Block",
    y = "Median RT (ms)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(
      size = 11,
      face = "plain"
    ),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    panel.spacing = unit(0.35, "lines"),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.key.width = unit(1.5, "lines"),
    plot.margin = margin(5, 5, 5, 5)
  )


ggsave(
  file.path(out_dir, "Supplementary_Fig9.pdf"),
  FigS9,
  width = 12,
  height = 3.6,
  units = "in"
)

ggsave(
  file.path(out_dir, "Supplementary_Fig9.png"),
  FigS9,
  width = 12,
  height = 3.6,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Supplementary_Fig9.svg"),
  FigS9,
  width = 12,
  height = 3.6,
  units = "in"
)


# ============================================================
# SUPPLEMENTARY FIGURE 10
# Hit rate by cluster, block, and cue magnitude
# ============================================================

s10 <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 10"
) %>%
  mutate(
    cluster = factor(cluster, levels = c(1, 2, 3)),
    block = as.numeric(block),
    cue = factor(
      cue,
      levels = c(
        "neutral",
        "small_reward",
        "medium_reward",
        "large_reward"
      ),
      labels = c(
        "Neutral",
        "Small reward",
        "Medium reward",
        "Large reward"
      )
    ),
    hit_percent = hit_rate * 100
  )

# Mean +/- SE across participants for each cluster x block x cue
s10_summary <- s10 %>%
  group_by(cluster, block, cue) %>%
  summarise(
    mean_hit = mean(hit_percent, na.rm = TRUE),
    se_hit = sd(hit_percent, na.rm = TRUE) /
      sqrt(sum(!is.na(hit_percent))),
    .groups = "drop"
  )

FigS10 <- ggplot(
  s10_summary,
  aes(
    x = factor(block),
    y = mean_hit,
    fill = cluster
  )
) +
  geom_col(
    position = position_dodge(width = 0.78),
    width = 0.70
  ) +
  geom_errorbar(
    aes(
      ymin = mean_hit - se_hit,
      ymax = mean_hit + se_hit
    ),
    position = position_dodge(width = 0.78),
    width = 0.14,
    linewidth = 0.45
  ) +
  facet_wrap(
    ~cue,
    nrow = 1
  ) +
  scale_fill_manual(
    values = primary_cluster_cols,
    labels = cluster_labels_s9s10,
    name = NULL
  ) +
  scale_y_continuous(
    breaks = seq(0, 100, 20),
    expand = expansion(mult = c(0, 0.03))
  ) +
  coord_cartesian(
    ylim = c(0, 100)
  ) +
  labs(
    x = "Block",
    y = "Hit rate (%)"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    strip.text = element_text(
      size = 11,
      face = "plain"
    ),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 9),
    panel.spacing = unit(0.35, "lines"),
    legend.position = "bottom",
    legend.text = element_text(size = 9),
    legend.key.width = unit(1.5, "lines"),
    plot.margin = margin(5, 5, 5, 5)
  )


ggsave(
  file.path(out_dir, "Supplementary_Fig10.pdf"),
  FigS10,
  width = 12,
  height = 3.6,
  units = "in"
)

ggsave(
  file.path(out_dir, "Supplementary_Fig10.png"),
  FigS10,
  width = 12,
  height = 3.6,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Supplementary_Fig10.svg"),
  FigS10,
  width = 12,
  height = 3.6,
  units = "in"
)

# ============================================================
# SUPPLEMENTARY FIGURE 16
# ROC curves: Placebo and All sample
# ============================================================

s16 <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 16"
) %>%
  mutate(
    sample = factor(
      sample,
      levels = c("placebo", "all"),
      labels = c("Placebo", "All sample")
    ),
    predictor = factor(
      predictor,
      levels = c("shift", "cue", "outcome"),
      labels = c(
        "Reward-to-cue SHIFT",
        "CUE only",
        "OUTCOME only"
      )
    )
  )

roc_cols <- c(
  "Reward-to-cue SHIFT" = "#2C86C5",
  "CUE only" = "darkgreen",
  "OUTCOME only" = "indianred"
)

# One AUC value per sample x predictor for legend labels
s16_auc <- s16 %>%
  distinct(sample, predictor, AUC) %>%
  mutate(
    legend_label = paste0(
      predictor,
      "   AUC = ",
      sprintf("%.2f", AUC)
    )
  )

s16 <- s16 %>%
  left_join(
    s16_auc %>% select(sample, predictor, legend_label),
    by = c("sample", "predictor")
  )

roc_label_cols <- setNames(
  roc_cols[as.character(s16_auc$predictor)],
  s16_auc$legend_label
)

plot_roc_panel <- function(data, panel_title) {
  
  ggplot(
    data,
    aes(
      x = FPR,
      y = TPR,
      color = legend_label,
      group = predictor
    )
  ) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      linewidth = 0.6,
      color = "grey65"
    ) +
    geom_step(
      linewidth = 1.15,
      direction = "hv"
    ) +
    
    # Youden index annotation for the reward-to-cue SHIFT curve
    annotate(
      "point",
      x = ifelse(panel_title == "Placebo", 0.35, 0.70),
      y = ifelse(panel_title == "Placebo", 0.82, 0.87),
      shape = 23,
      size = 3.2,
      stroke = 0.9,
      fill = "white",
      color = "#2C86C5"
    ) +
    annotate(
      "text",
      x = ifelse(panel_title == "Placebo", 0.15, 0.50),
      y = ifelse(panel_title == "Placebo", 0.85, 0.90),
      label = ifelse(
        panel_title == "Placebo",
        "Youden\nθ = 0.05",
        "Youden\nθ = 0.64"
      ),
      hjust = 0,
      vjust = 1,
      size = 2.6,
      color = "#2C86C5"
    ) +
    
    scale_color_manual(
      values = roc_label_cols,
      name = NULL
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.2),
      expand = c(0, 0)
    ) +
    coord_fixed(ratio = 1) +
    labs(
      title = panel_title,
      x = "1 − Specificity (False Positive Rate)",
      y = "Sensitivity (True Positive Rate)"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(
        size = 15,
        face = "bold",
        hjust = 0.5
      ),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 9),
      panel.grid.minor = element_blank(),
      legend.position = c(0.74, 0.16),
      legend.justification = c(0.5, 0),
      legend.background = element_rect(
        fill = "white",
        color = "grey80",
        linewidth = 0.3
      ),
      legend.text = element_text(size = 7.5),
      legend.key.width = unit(1.3, "lines"),
      plot.margin = margin(8, 8, 8, 8)
    )
}

p_S16_placebo <- plot_roc_panel(
  s16 %>% filter(sample == "Placebo"),
  "Placebo"
)

p_S16_all <- plot_roc_panel(
  s16 %>% filter(sample == "All sample"),
  "All sample"
)

FigS16 <-
  p_S16_placebo + p_S16_all +
  plot_layout(ncol = 2)


ggsave(
  file.path(out_dir, "Supplementary_Fig16.pdf"),
  FigS16,
  width = 10,
  height = 4.8,
  units = "in"
)

ggsave(
  file.path(out_dir, "Supplementary_Fig16.png"),
  FigS16,
  width = 10,
  height = 4.8,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Supplementary_Fig16.svg"),
  FigS16,
  width = 10,
  height = 4.8,
  units = "in"
)

# ============================================================
# SUPPLEMENTARY FIGURE 17
# Block-wise predictability of cluster membership
# Final source data: three separate sheets (17a, 17b, 17c)
# ============================================================

# ------------------------------------------------------------
# Panel A source data
# ------------------------------------------------------------

s17a <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 17a"
) %>%
  mutate(
    predictor = factor(
      predictor,
      levels = c(
        "Block1", "Block2", "Block3", "Block4",
        "Early", "Later", "Overall", "Ortho"
      ),
      labels = c(
        "Block 1", "Block 2", "Block 3", "Block 4",
        "Early\nchange", "Later\nchange",
        "Overall\nchange", "Orthogonal\nchange"
      )
    )
  )

# ------------------------------------------------------------
# Panel A1: Cross-validated accuracy
# ------------------------------------------------------------

best_acc <- max(s17a$accuracy, na.rm = TRUE)

p_S17_acc <- ggplot(
  s17a,
  aes(x = predictor, y = accuracy, group = 1)
) +
  geom_col(
    aes(fill = accuracy == best_acc),
    width = 0.70,
    color = "grey35",
    linewidth = 0.5
  ) +
  geom_line(
    color = "grey30",
    linewidth = 0.8
  ) +
  geom_point(
    color = "grey20",
    size = 2.7
  ) +
  geom_hline(
    yintercept = 1/3,
    linetype = "dotted",
    linewidth = 0.7,
    color = "grey55"
  ) +
  geom_hline(
    yintercept = best_acc,
    linetype = "dashed",
    linewidth = 0.7,
    color = "#FF3B30"
  ) +
  annotate(
    "text",
    x = 7.85,
    y = 1/3 + 0.015,
    label = "Chance (0.33)",
    hjust = 1,
    vjust = 0,
    size = 3.1,
    color = "grey45"
  ) +
  scale_fill_manual(
    values = c(
      "FALSE" = "#4C8DBD",
      "TRUE" = "#D13A2C"
    ),
    guide = "none"
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.25),
    expand = c(0, 0)
  ) +
  labs(
    title = "Cross-validated Accuracy",
    x = NULL,
    y = "Accuracy"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(
      size = 17,
      face = "plain",
      hjust = 0
    ),
    axis.title.y = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(5, 5, 4, 5)
  )

# ------------------------------------------------------------
# Panel A2: Multiclass AUC (Hand-Till)
# ------------------------------------------------------------

best_auc <- max(s17a$AUC_HT, na.rm = TRUE)

p_S17_auc <- ggplot(
  s17a,
  aes(x = predictor, y = AUC_HT, group = 1)
) +
  geom_col(
    aes(fill = AUC_HT == best_auc),
    width = 0.70,
    color = "grey35",
    linewidth = 0.5
  ) +
  geom_line(
    color = "grey30",
    linewidth = 0.8
  ) +
  geom_point(
    color = "grey20",
    size = 2.7
  ) +
  geom_hline(
    yintercept = 0.50,
    linetype = "dotted",
    linewidth = 0.7,
    color = "grey55"
  ) +
  geom_hline(
    yintercept = best_auc,
    linetype = "dashed",
    linewidth = 0.7,
    color = "#FF3B30"
  ) +
  annotate(
    "text",
    x = 7.85,
    y = 0.515,
    label = "Chance (0.50)",
    hjust = 1,
    vjust = 0,
    size = 3.1,
    color = "grey45"
  ) +
  scale_fill_manual(
    values = c(
      "FALSE" = "#2E86BE",
      "TRUE" = "#D13A2C"
    ),
    guide = "none"
  ) +
  scale_y_continuous(
    limits = c(0, 1.05),
    breaks = seq(0, 1, 0.25),
    expand = c(0, 0)
  ) +
  labs(
    title = "Multiclass AUC (Hand-Till)",
    x = NULL,
    y = "AUC"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(
      size = 17,
      face = "plain",
      hjust = 0
    ),
    axis.title.y = element_text(size = 12),
    axis.text = element_text(size = 10),
    panel.grid.minor = element_blank(),
    legend.position = "none",
    plot.margin = margin(4, 5, 5, 5)
  )

p_S17_A <-
  p_S17_acc /
  p_S17_auc +
  plot_annotation(
    #title = "Block-wise Predictability of Cluster Membership",
    #subtitle = "Primary MDD sample (N = 57)  |  Repeated 10-fold CV (5 repeats)  |  Multinomial classifier",
    tag_levels = "a"
  ) &
  theme(
    plot.annotation.title = element_text(
      size = 17,
      face = "bold"
    ),
    plot.annotation.subtitle = element_text(
      size = 11,
      color = "grey40"
    ),
    plot.tag = element_text(
      size = 15,
      face = "bold"
    )
  )

# ------------------------------------------------------------
# ROC helper for panels B and C
# IMPORTANT:
# Keep the deposited cluster values as C1/C2/C3 for grouping.
# Only the DISPLAY labels are changed to Cluster 1/2/3.
# ------------------------------------------------------------

cluster_roc_cols <- c(
  "C1" = "#440154FF",
  "C2" = "#22A884FF",
  "C3" = "#D55E00"
)

plot_S17_roc <- function(sheet_name, panel_title, panel_tag) {
  
  roc <- read_excel(
    source_file,
    sheet = sheet_name
  ) %>%
    mutate(
      cluster = factor(
        as.character(cluster),
        levels = c("C1", "C2", "C3")
      )
    )
  
  auc_labels <- roc %>%
    distinct(cluster, AUC) %>%
    mutate(
      cluster_display = recode(
        as.character(cluster),
        "C1" = "Cluster 1",
        "C2" = "Cluster 2",
        "C3" = "Cluster 3"
      ),
      legend_label = paste0(
        cluster_display,
        "  AUC = ",
        sprintf("%.2f", AUC)
      )
    )
  
  roc <- roc %>%
    left_join(
      auc_labels %>% select(cluster, AUC, legend_label),
      by = c("cluster", "AUC")
    )
  
  legend_cols <- setNames(
    cluster_roc_cols[as.character(auc_labels$cluster)],
    auc_labels$legend_label
  )
  
  ggplot(
    roc,
    aes(
      x = FPR,
      y = TPR,
      color = legend_label,
      group = cluster
    )
  ) +
    geom_abline(
      intercept = 0,
      slope = 1,
      linetype = "dashed",
      linewidth = 0.6,
      color = "grey70"
    ) +
    geom_step(
      linewidth = 1.15,
      direction = "hv",
      na.rm = FALSE
    ) +
    scale_color_manual(
      values = legend_cols,
      name = NULL
    ) +
    scale_x_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.25),
      expand = c(0, 0)
    ) +
    scale_y_continuous(
      limits = c(0, 1),
      breaks = seq(0, 1, 0.25),
      expand = c(0, 0)
    ) +
    coord_fixed(ratio = 1) +
    labs(
      title = panel_title,
      x = "1 - Specificity",
      y = "Sensitivity",
      tag = panel_tag
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(
        size = 12,
        face = "plain",
        hjust = 0.5
      ),
      plot.tag = element_text(
        size = 15,
        face = "bold"
      ),
      axis.title = element_text(size = 10),
      axis.text = element_text(size = 8.5),
      panel.grid.minor = element_blank(),
      legend.position = c(0.68, 0.18),
      legend.background = element_rect(
        fill = "white",
        color = "grey80",
        linewidth = 0.3
      ),
      legend.text = element_text(size = 7.5),
      legend.key.width = unit(1.1, "lines"),
      plot.margin = margin(5, 8, 5, 8)
    )
}

# ------------------------------------------------------------
# Panel B: Block 4 ROC
# ------------------------------------------------------------

p_S17_B <- plot_S17_roc(
  "Supplementary Fig. 17b",
  "Block 4",
  "b"
)

# ------------------------------------------------------------
# Panel C: Overall change ROC
# ------------------------------------------------------------

p_S17_C <- plot_S17_roc(
  "Supplementary Fig. 17c",
  "Overall change",
  "c"
)

# ------------------------------------------------------------
# Final layout
# ------------------------------------------------------------

FigS17 <-
  p_S17_A /
  (p_S17_B + p_S17_C) +
  plot_layout(
    heights = c(1.5, 1.5, 2.5)
  )


ggsave(
  file.path(out_dir, "Supplementary_Fig17.pdf"),
  FigS17,
  width = 11,
  height = 11,
  units = "in"
)

ggsave(
  file.path(out_dir, "Supplementary_Fig17.png"),
  FigS17,
  width = 10,
  height = 11,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Supplementary_Fig17.svg"),
  FigS17,
  width = 10,
  height = 11,
  units = "in"
)


# ============================================================
# SUPPLEMENTARY FIGURE 18
# Posterior probabilities for cluster assignment
# ============================================================

figS18 <- read_excel(
  source_file,
  sheet = "Supplementary Fig. 18"
) %>%
  mutate(
    cluster_pred = factor(cluster_pred, levels = c("C1", "C2", "C3")),
    row_index = as.numeric(row_index),
    shift_accumbens = as.numeric(shift_accumbens),
    C1 = as.numeric(C1),
    C2 = as.numeric(C2),
    C3 = as.numeric(C3)
  )

# Same cluster colors used throughout the manuscript,
# renamed to match the labels used in this figure
cluster_cols_S18 <- c(
  "Cluster 1" = unname(primary_cluster_cols[1]),
  "Cluster 2" = unname(primary_cluster_cols[2]),
  "Cluster 3" = unname(primary_cluster_cols[3])
)

# Long format for posterior probabilities
figS18_long <- figS18 %>%
  select(
    row_index,
    ID,
    shift_accumbens,
    cluster_pred,
    C1,
    C2,
    C3
  ) %>%
  pivot_longer(
    cols = c(C1, C2, C3),
    names_to = "cluster_probability",
    values_to = "posterior_probability"
  ) %>%
  mutate(
    Cluster = factor(
      cluster_probability,
      levels = c("C1", "C2", "C3"),
      labels = c("Cluster 1", "Cluster 2", "Cluster 3")
    )
  )


# ------------------------------------------------------------
# Panel A: Distribution of posterior probabilities
# ------------------------------------------------------------

p_S18A <- ggplot(
  figS18_long,
  aes(
    x = posterior_probability,
    fill = Cluster,
    color = Cluster
  )
) +
  geom_density(
    alpha = 0.35,
    linewidth = 0.8,
    adjust = 1
  ) +
  scale_fill_manual(values = cluster_cols_S18) +
  scale_color_manual(values = cluster_cols_S18) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, 0.25),
    expand = c(0, 0)
  ) +
  labs(
    x = "Posterior probability",
    y = "Density",
    fill = "Cluster",
    color = "Cluster",
    tag = "a"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 15
    ),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7.5),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 8.5),
    plot.margin = margin(5, 5, 5, 5)
  )


# ------------------------------------------------------------
# Panel B: Subject-level posterior-probability heatmap
# ------------------------------------------------------------

p_S18B <- ggplot(
  figS18_long,
  aes(
    x = Cluster,
    y = row_index,
    fill = posterior_probability
  )
) +
  geom_tile() +
  scale_fill_gradient(
    low = "white",
    high = "#2C7FB8",
    limits = c(0, 1),
    name = "prob"
  ) +
  scale_y_continuous(
    breaks = seq(0, 100, 25),
    expand = c(0, 0)
  ) +
  labs(
    x = "Cluster",
    y = "Subjects",
    tag = "b"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 15
    ),
    panel.grid = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7.5),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 8.5),
    plot.margin = margin(5, 5, 5, 5)
  )


# ------------------------------------------------------------
# Panel C:
# Posterior probability as a function of reward-to-cue shift
# ------------------------------------------------------------

p_S18C <- ggplot(
  figS18_long,
  aes(
    x = shift_accumbens,
    y = posterior_probability,
    color = Cluster
  )
) +
  geom_point(
    size = 1.25,
    alpha = 0.45
  ) +
  geom_smooth(
    aes(group = Cluster),
    method = "loess",
    formula = y ~ x,
    se = FALSE,
    linewidth = 1.1,
    span = 0.75
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    linewidth = 0.5,
    color = "grey60"
  ) +
  scale_color_manual(
    values = cluster_cols_S18
  ) +
  scale_y_continuous(
    limits = c(-0.05, 1),
    breaks = seq(0, 1, 0.25),
    expand = c(0, 0)
  ) +
  labs(
    x = "Reward-to-cue shift\nNAcc BOLD Beta weights",
    y = "Posterior probability",
    color = "Cluster",
    tag = "c"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.tag = element_text(
      face = "bold",
      size = 15
    ),
    panel.grid.minor = element_blank(),
    legend.position = "right",
    legend.title = element_text(size = 8),
    legend.text = element_text(size = 7.5),
    axis.title = element_text(size = 10),
    axis.text = element_text(size = 8.5),
    plot.margin = margin(5, 5, 5, 5)
  )


# ------------------------------------------------------------
# Assemble Supplementary Figure 18
# ------------------------------------------------------------

FigS18 <-
  p_S18A +
  p_S18B +
  p_S18C +
  plot_layout(
    ncol = 3,
    widths = c(1.08, 1.05, 1.12)
  )



# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

ggsave(
  file.path(out_dir, "Supplementary_Fig18.pdf"),
  FigS18,
  width = 15,
  height = 4.4,
  units = "in"
)

ggsave(
  file.path(out_dir, "Supplementary_Fig18.png"),
  FigS18,
  width = 15,
  height = 4.4,
  units = "in",
  dpi = 600
)

ggsave(
  file.path(out_dir, "Supplementary_Fig18.svg"),
  FigS18,
  width = 15,
  height = 4.4,
  units = "in"
)


message("Finished R panels. Files are in: ",out_dir)
