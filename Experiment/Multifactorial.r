# Se instalan los paquetes necesarios
if (!require(psych)) {
    install.packages("psych")
}
if (!require(FSA)) {
    install.packages("FSA")
}
if (!require(ggplot2)) {
    install.packages("ggplot2")
}
if (!require(car)) {
    install.packages("car")
}
if (!require(multcompView)) {
    install.packages("multcompView")
}
if (!require(multcomp)) {
    install.packages("multcomp")
}
if (!require(emmeans)) {
    install.packages("emmeans")
}
if (!require(lsmeans)) {
    install.packages("lsmeans")
}
if (!require(rcompanion)) {
    install.packages("rcompanion")
}
if (!require(Rmisc)) {
    installed.packages("Rmisc")
}
if (!require(rcompanion)) {
    installed.packages("phia")
}

library(FSA)
library(rcompanion)
library(ggplot2)
library(car)
library(multcompView)
library(lsmeans)
library(multcomp)
library(phia)

datos <- ("
algorithm seed time result
mcts 6 2 138.8359341621399
rhea 2 3 118.76265287399292
rhea 4 3 127.463219165802
mcts 7 2 145.22519063949585
mcts 2 2 229.5522108078003
mcts 1 3 72.57336759567261
mcts 5 3 63.69783163070679
mcts 0 3 69.36414766311646
rhea 1 2 58.0353627204895
rhea 3 2 105.24952220916748
rhea 7 3 93.86166381835938
mcts 3 3 144.35660195350647
mcts 1 2 59.203301429748535
rhea 7 2 70.01967811584473
mcts 4 3 147.4343113899231
rhea 1 3 54.652929067611694
mcts 2 3 162.8265461921692
mcts 7 3 138.9909644126892
mcts 0 2 57.40394067764282
rhea 0 2 58.98468995094299
rhea 5 3 49.93284559249878
rhea 0 3 61.12894368171692
mcts 3 2 149.6534173488617
mcts 6 3 191.7427363395691
rhea 2 2 110.51442909240723
rhea 6 3 101.24994277954102
mcts 4 2 124.25028657913208
rhea 6 2 101.65261459350586
rhea 5 2 53.04967427253723
rhea 3 3 168.67461895942688
rhea 4 2 98.38874411582947
mcts 5 2 58.07075309753418

rhea 7 2 108.6063597202301
rhea 5 3 35.69115614891052
mcts 1 3 77.9771180152893
mcts 3 3 122.55584073066711
rhea 2 2 106.44907450675964
rhea 1 3 52.74167346954346
mcts 7 2 99.5571665763855
rhea 3 3 107.258296251297
rhea 4 2 94.69349694252014
rhea 1 2 55.89291763305664
mcts 4 3 168.48306107521057
mcts 2 3 167.13138484954834
mcts 6 2 151.28718209266663
rhea 7 3 114.47497797012329
mcts 7 3 181.43094491958618
rhea 3 2 107.03808307647705
rhea 4 3 127.32624077796936
rhea 5 2 47.360020875930786
mcts 4 2 120.17152571678162
rhea 6 3 101.1959719657898
mcts 5 2 55.711244106292725
mcts 0 2 89.22618007659912
rhea 0 3 56.904011726379395
mcts 5 3 87.25414037704468
rhea 2 3 135.35331010818481
rhea 0 2 61.44936919212341
mcts 6 3 143.3251929283142
mcts 3 2 162.9954378604889
mcts 0 3 71.03875589370728
rhea 6 2 148.63651084899902
mcts 1 2 68.694983959198
mcts 2 2 143.4628975391388

mcts 5 2 56.50719404220581
mcts 1 3 77.35775566101074
mcts 2 2 130.55824494361877
mcts 4 2 117.10759496688843
rhea 2 2 109.98535442352295
rhea 7 2 128.2041254043579
rhea 3 3 101.8457396030426
rhea 4 3 159.30690836906433
rhea 0 2 60.71700930595398
mcts 2 3 206.09032320976257
rhea 3 2 184.31933426856995
rhea 7 3 124.45530247688293
rhea 6 3 102.8815565109253
rhea 5 3 47.44490885734558
rhea 1 3 59.8391318321228
mcts 4 3 158.83681726455688
mcts 7 2 170.91653871536255
mcts 0 3 89.62203454971313
mcts 7 3 165.77371430397034
rhea 6 2 105.60842061042786
mcts 3 2 114.67203521728516
rhea 4 2 77.33748483657837
rhea 5 2 66.428457736969
rhea 0 3 49.18642044067383
mcts 6 2 188.83019924163818
mcts 6 3 141.85840845108032
mcts 5 3 74.77298069000244
mcts 0 2 131.53040480613708
mcts 1 2 81.18489956855774
mcts 3 3 115.9804916381836
rhea 2 3 133.95507287979126
rhea 1 2 66.61156964302063

rhea 7 2 96.29975056648254
rhea 4 2 104.32942008972168
rhea 2 2 107.48337483406067
rhea 1 2 65.83313751220703
mcts 3 2 174.49970865249634
mcts 1 2 94.84573698043823
rhea 6 2 138.83185625076294
rhea 3 2 105.8466808795929
rhea 0 2 54.99426555633545
mcts 6 2 133.63238978385925
mcts 3 3 189.02281427383423
mcts 5 2 58.03930640220642
mcts 0 2 51.766671895980835
rhea 2 3 108.63765811920166
rhea 6 3 120.28362727165222
mcts 2 2 145.00633692741394
rhea 4 3 110.87477040290833
mcts 5 3 64.62694096565247
mcts 1 3 110.9574773311615
mcts 4 2 166.4997754096985
rhea 0 3 51.61681246757507
rhea 3 3 116.59962177276611
rhea 5 2 47.94415283203125
mcts 7 3 103.50526237487793
mcts 4 3 154.31737613677979
rhea 1 3 52.99504637718201
mcts 2 3 223.3491952419281
mcts 0 3 80.1494071483612
mcts 7 2 198.0874457359314
rhea 7 3 128.46753072738647
mcts 6 3 164.10403680801392
rhea 5 3 46.80270743370056

rhea 2 3 165.7538764476776
mcts 3 2 138.72960090637207
mcts 1 2 79.0133626461029
mcts 2 3 163.0355999469757
mcts 7 2 116.72412514686584
mcts 0 2 69.12873792648315
rhea 1 2 112.56045508384705
rhea 1 3 52.489279985427856
rhea 0 2 57.34534978866577
mcts 1 3 78.575190782547
rhea 5 2 49.88945770263672
rhea 0 3 54.58628296852112
rhea 6 2 102.02368497848511
rhea 4 3 96.42895293235779
rhea 7 2 114.07295989990234
mcts 6 3 218.23927974700928
rhea 2 2 136.72870659828186
mcts 4 2 127.27581214904785
mcts 6 2 150.94331693649292
rhea 3 3 85.88147354125977
mcts 5 2 53.99145221710205
mcts 3 3 180.6466314792633
rhea 5 3 53.271480321884155
rhea 6 3 124.38660168647766
mcts 5 3 63.5389838218689
mcts 0 3 98.62924766540527
rhea 4 2 91.53003478050232
rhea 7 3 125.14301490783691
mcts 4 3 135.36774849891663
mcts 2 2 169.8681173324585
rhea 3 2 93.41802859306335
mcts 7 3 169.60089874267578
")

# SE GENERA LA TABLA CON LOS DATOS Y SE ELIMINAN LOS DATOS CARGADOS
# DESDE EL ARCHIVO
data <- read.table(textConnection(datos), header = TRUE)

rm(datos)

#SE ORDENAN LOS DATOS
data$seed <- factor(data$seed, levels = unique(data$seed))
data$time <- factor(data$time, levels = unique(data$time))
data$algorithm <- factor(data$algorithm, levels = unique(data$algorithm))


#VERIFICAMOS QUE TODO ESTE BIEN
headtail(data)
summary(data)
str(data)

#SE GENERA EL GRAFICO SIMPLE DE INTERACCION
interaction.plot(x.factor = data$seed,
                 trace.factor = data$algorithm,
                 response = data$result,
                 fun = mean,
                 type = "b",
                 col = c("black", "red", "green"),
                 pch = c(19, 17, 15),
                 fixed = TRUE,
                 leg_bty = "o")

interaction.plot(x.factor = data$time,
                 trace.factor = data$algorithm,
                 response = data$result,
                 fun = mean,
                 type = "b",
                 col = c("black", "red", "green"),
                 pch = c(19, 17, 15),
                 fixed = TRUE,
                 leg_bty = "o")

# SE GENERA EL MODELO LINEAL DE LOS DATOS Y EL ANALISIS ANOVA
# model <- lm(result ~ seed + algorithm + seed:algorithm, data = data) # nolint
model <- lm(result ~ seed * algorithm * time, data = data)

Anova(model, type = "II")

# SE EVALUAN LOS SUPUESTOS DESPUES DE OBTENER EL MODELO LINEAL
# SE GENERA EL HISTOGRAMA DE RESIDUOS
x <- residuals(model)
plotNormalHistogram(x)

plot(fitted(model), residuals(model))
plot(model)

# SE HACE LA COMPARACION DE PARES DE LOS PROMEDIOS DE MINIMOS CUADRADOS
# DE CADA UNO DE LOS GRUPOS
marginal <- lsmeans(model, ~ algorithm)
pairs(marginal, adjust = "tukey")

# SE IDENTIFICAN LOS GRUPOS DIFERENTES EN LOS QUE SE CLASIFICAN LOS ALGORITMOS
# EN ESTUDIO
cld <- cld(marginal, alpha = 0.05, Letters = letters, adjust = "tukey")
cld

# seed
marginal <- lsmeans(model, ~ seed)
pairs(marginal, adjust = "tukey")

# SE IDENTIFICAN LOS GRUPOS DIFERENTES EN LOS QUE SE CLASIFICAN LOS seed
# EN ESTUDIO
cld <- cld(marginal, alpha = 0.05, Letters = letters, adjust = "tukey")
cld

# time
marginal <- lsmeans(model, ~ time)
pairs(marginal, adjust = "tukey")

# SE IDENTIFICAN LOS GRUPOS DIFERENTES EN LOS QUE SE CLASIFICAN LOS time
# EN ESTUDIO
cld <- cld(marginal, alpha = 0.05, Letters = letters, adjust = "tukey")
cld

# Summarize mean result vs algorithm
sum <- Summarize(result ~ algorithm, data = data, digits = 3)

sum$se <- sum$sd / sqrt(sum$n)
sum$se <- signif(sum$se, digits = 3)
sum

sum$algorithm <- factor(sum$algorithm, levels = unique(sum$algorithm))

#GRAFICO
pd <- position_dodge(.2)

ggplot(sum, aes(x = algorithm,
                y = mean,
                color = algorithm)) +
  geom_errorbar(aes(ymin = mean - se,
                    ymax = mean + se,
                    width = 0.2,
                    size = 0.7)) +
  geom_point(shape = 15, size = 4, position = pd) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold")) +
  scale_colour_manual(values = c("black", "red", "green")) +
  ylab("result")

# Summarize mean result vs seed in function of the algorithm
sum <- Summarize(result ~ seed + algorithm, data = data, digits = 3)

sum$se <- sum$sd / sqrt(sum$n)
sum$se <- signif(sum$se, digits = 3)
sum

sum$seed <- factor(sum$seed, levels = unique(sum$seed))

#GRAFICO
pd <- position_dodge(.2)

ggplot(sum, aes(x = seed,
                y = mean,
                color = algorithm)) +
  geom_errorbar(aes(ymin = mean - se,
                    ymax = mean + se,
                    width = 0.2,
                    size = 0.7)) +
  geom_point(shape = 15, size = 4, position = pd) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold")) +
  scale_colour_manual(values = c("black", "red", "green")) +
                        ylab("result")


# Summaruze mean result vs time in function of the algorithm
sum <- Summarize(result ~ time + algorithm, data = data, digits = 3)

sum$se <- sum$sd / sqrt(sum$n)
sum$se <- signif(sum$se, digits = 3)
sum

sum$time <- factor(sum$time, levels = unique(sum$time))

#GRAFICO
pd <- position_dodge(.2)

ggplot(sum, aes(x = time,
                y = mean,
                color = algorithm)) +
  geom_errorbar(aes(ymin = mean - se,
                    ymax = mean + se,
                    width = 0.2,
                    size = 0.7)) +
  geom_point(shape = 15, size = 4, position = pd) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold")) +
  scale_colour_manual(values = c("black", "red", "green")) +
                        ylab("result")


