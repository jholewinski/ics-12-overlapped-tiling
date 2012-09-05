# Plot scripts for GPU models

require('ggplot2')

plot_real_vs_sim <- function(data, name='Real vs Model') {
  graph_data <- data.frame(data, config=1:nrow(data))
  ggplot(graph_data, aes(x=config)) +
#    geom_line(aes(y=real_elapsed, color="Real"), size=1) +
#    geom_point(aes(y=real_elapsed, color="Real"), size=3, shape=16) +
    geom_line(aes(y=sim_elapsed, color="Sim"), linetype=2, size=1) +
    geom_point(aes(y=sim_elapsed, color="Sim"), size=3, shape=17) +
#    geom_line(aes(y=sim_elapsed_upper, color="Sim (upper)"), linetype=3, size=1) +
#    geom_point(aes(y=sim_elapsed_upper, color="Sim (upper)"), size=3, shape=18) +
    geom_line(aes(y=event_elapsed, color="Real"), linetype=4, size=1) +
    geom_point(aes(y=event_elapsed, color="Real"), size=3, shape=19) +
    xlab('Configuration') +
    ylab('Elapsed Time (sec)') +
    opts(legend.title=theme_blank()) +
    opts(title=name)
}

gen_clocks_table <- function(data) {
  data.frame(data,
             p1_clocks=data$clock1-data$clock0,
             p2_clocks=data$clock2-data$clock1,
             p3_clocks=data$clock3-data$clock2,
             p4_clocks=data$clock4-data$clock3)
}