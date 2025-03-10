---
title: "U.S. Cities"
author: Eduardo Ramirez

editor: visual
execute:
  keep-md: true

date: "November 04, 2024"
warnings: false
format:
  html:
    df-print: paged
    code-fold: true
    code-line-numbers: true
---


::: {.cell}

```{.r .cell-code}
if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, sf, USAboundaries, ggrepel)



states_data <- us_states() |> 
  filter(!state_abbr %in% c("AK", "HI"))

idaho_counties <- us_counties(states = "Idaho")

# get city data
cities_data <- us_cities()

top_3_cities <- cities_data |>
  filter(!state_name %in% c("Hawaii", "Alaska", "Puerto Rico")) |>
  group_by(state_name) |>
  arrange(desc(population)) |>
  slice_head(n = 3) |>
  mutate(rank = row_number(), population = population / 1000) |>
  ungroup()

coords <- st_coordinates(top_3_cities$geometry)
top_3_cities <- top_3_cities |>
  mutate(longitude = coords[, 1], latitude = coords[, 2])

largest_city_per_state <- top_3_cities |>
  filter(rank == 1)

# Plot the map
ggplot() +
  geom_sf(data = states_data, fill = "NA", color = "grey50") +
  geom_sf(data = idaho_counties, fill = 'NA', color = 'black') +
  geom_point(data = top_3_cities, 
             aes(x = longitude, y = latitude, size = population, color = factor(rank))) +
  geom_text_repel(data = largest_city_per_state, aes(x = longitude, y = latitude, label = city), 
                  size = 3, 
                  color = "black" , 
                  box.padding = 0.3, 
                  point.padding = 0.5, 
                  segment.color = 'grey50',
                  max.overlaps = 50) +
  theme_bw() +
  labs(size = 'Population \n (1,000)', x ='', y = '') +
  theme(legend.position = "right",
        legend.title = element_text(size = 12),
        legend.text = element_text(size = 10)) +
  guides(color = FALSE)
```

::: {.cell-output-display}
![](US_cities_28_files/figure-html/testing-1.png){width=960}
:::

```{.r .cell-code}
# Save the plot as a .png file
ggsave("usa_top_3_cities_map.png", width = 15, height = 10)
```
:::
