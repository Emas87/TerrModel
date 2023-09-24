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
mcts 6 2 156.5691318511963
rhea 2 3 219.79939532279968
rhea 4 3 137.04501032829285
mcts 7 2 133.19303178787231
mcts 2 2 237.82162618637085
mcts 1 3 92.37173366546631
mcts 5 3 111.39669609069824
mcts 0 3 113.66165399551392
rhea 1 2 129.74975299835205
rhea 3 2 172.64069414138794
rhea 7 3 200.87978672981262
mcts 3 3 210.81960153579712
mcts 1 2 152.8854067325592
rhea 7 2 140.8988151550293
mcts 4 3 158.6687047481537
rhea 1 3 100.93246746063232
mcts 2 3 214.41954398155212
mcts 7 3 225.20479226112366
mcts 0 2 108.40998649597168
rhea 0 2 82.8861756324768
rhea 5 3 131.68603110313416
rhea 0 3 198.0907073020935
mcts 3 2 115.17571496963501
mcts 6 3 204.82116770744324
rhea 2 2 199.31339383125305
rhea 6 3 185.3601050376892
mcts 4 2 147.72477841377258
rhea 6 2 175.49163126945496
rhea 5 2 80.42252230644226
rhea 3 3 202.83732748031616
rhea 4 2 133.01491355895996
mcts 5 2 85.98520922660828

rhea 7 2 194.9495792388916
rhea 5 3 105.23713850975037
mcts 1 3 269.59328413009644
mcts 3 3 290.3000576496124
rhea 2 2 155.57732558250427
rhea 1 3 111.82768869400024
mcts 7 2 169.84512186050415
rhea 3 3 190.31409335136414
rhea 4 2 179.60209321975708
rhea 1 2 209.73421096801758
mcts 4 3 134.57412314414978
mcts 2 3 192.6369388103485
mcts 6 2 202.83350038528442
rhea 7 3 193.40499687194824
mcts 7 3 299.17926955223083
rhea 3 2 155.85977339744568
rhea 4 3 234.50029158592224
rhea 5 2 102.11136507987976
mcts 4 2 125.79469752311707
rhea 6 3 171.0053699016571
mcts 5 2 81.74409627914429
mcts 0 2 95.92624282836914
rhea 0 3 120.26160955429077
mcts 5 3 105.66843461990356
rhea 2 3 208.1565647125244
rhea 0 2 98.21877694129944
mcts 6 3 250.00326681137085
mcts 3 2 145.16448855400085
mcts 0 3 147.6749963760376
rhea 6 2 151.8443717956543
mcts 1 2 104.39142894744873
mcts 2 2 213.52396774291992

mcts 5 2 76.33618831634521
mcts 1 3 93.87201619148254
mcts 2 2 198.2295002937317
mcts 4 2 190.98475980758667
rhea 2 2 155.02112793922424
rhea 7 2 198.3829483985901
rhea 3 3 233.9864547252655
rhea 4 3 138.8304727077484
rhea 0 2 88.82187485694885
mcts 2 3 205.19269132614136
rhea 3 2 156.33839583396912
rhea 7 3 203.27343440055847
rhea 6 3 163.91991806030273
rhea 5 3 190.52193021774292
rhea 1 3 271.2757155895233
mcts 4 3 139.4194142818451
mcts 7 2 190.94598984718323
mcts 0 3 141.46428894996643
mcts 7 3 216.09912729263306
rhea 6 2 132.94512248039246
mcts 3 2 222.63425135612488
rhea 4 2 113.63795757293701
rhea 5 2 96.58176565170288
rhea 0 3 106.5785493850708
mcts 6 2 168.56324005126953
mcts 6 3 275.77434611320496
mcts 5 3 105.78111124038696
mcts 0 2 109.75903367996216
mcts 1 2 187.40017008781433
mcts 3 3 272.02826380729675
rhea 2 3 149.42189526557922
rhea 1 2 123.57811141014099

rhea 7 2 235.67356514930725
rhea 4 2 130.38885712623596
rhea 2 2 208.22643876075745
rhea 1 2 134.08526182174683
mcts 3 2 203.84134793281555
mcts 1 2 86.87339544296265
rhea 6 2 164.8029489517212
rhea 3 2 148.54610633850098
rhea 0 2 89.9444808959961
mcts 6 2 176.29947662353516
mcts 3 3 211.4588303565979
mcts 5 2 85.39666700363159
mcts 0 2 93.15916991233826
rhea 2 3 213.3443169593811
rhea 6 3 194.93065428733826
mcts 2 2 185.44292998313904
rhea 4 3 133.7801263332367
mcts 5 3 116.57391095161438
mcts 1 3 103.97783398628235
mcts 4 2 145.07135128974915
rhea 0 3 90.41676759719849
rhea 3 3 167.9379997253418
rhea 5 2 80.92302536964417
mcts 7 3 242.94859290122986
mcts 4 3 138.54184675216675
rhea 1 3 253.0996437072754
mcts 2 3 224.25706887245178
mcts 0 3 197.65769028663635
mcts 7 2 190.08676433563232
rhea 7 3 206.6653699874878
mcts 6 3 172.51871490478516
rhea 5 3 79.46983098983765

rhea 2 3 189.52651596069336
mcts 3 2 148.15884041786194
mcts 1 2 77.33760476112366
mcts 2 3 240.22930884361267
mcts 7 2 226.70120453834534
mcts 0 2 185.06620144844055
rhea 1 2 96.56639742851257
rhea 1 3 110.15878057479858
rhea 0 2 94.50266790390015
mcts 1 3 116.56413650512695
rhea 5 2 82.37934136390686
rhea 0 3 88.95668649673462
rhea 6 2 161.99786019325256
rhea 4 3 142.7427065372467
rhea 7 2 146.31722855567932
mcts 6 3 209.3561658859253
rhea 2 2 198.16097044944763
mcts 4 2 123.77343535423279
mcts 6 2 167.63202810287476
rhea 3 3 202.8498661518097
mcts 5 2 92.61046767234802
mcts 3 3 215.2208173274994
rhea 5 3 145.99872374534607
rhea 6 3 209.8558051586151
mcts 5 3 136.8423719406128
mcts 0 3 140.4361367225647
rhea 4 2 117.0086190700531
rhea 7 3 180.3333079814911
mcts 4 3 235.4303650856018
mcts 2 2 191.67655062675476
rhea 3 2 170.74978041648865
mcts 7 3 274.7594225406647
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
model <- lm(result ~ algorithm * seed * time, data = data)
Anova(model, type = "II", alpha = 0.05)

# SE EVALUAN LOS SUPUESTOS DESPUES DE OBTENER EL MODELO LINEAL
# SE GENERA EL HISTOGRAMA DE RESIDUOS para verificar normalidad
x <- residuals(model)
plotNormalHistogram(x)

# Homocedasticidad
plot(fitted(model), residuals(model))

plot(model)

############################################################################################################################# nolint
# Prueba de Levene
leveneTest(result ~ algorithm * seed * time, data = data)


# POST-HOC
# SE HACE LA COMPARACION DE PARES DE LOS PROMEDIOS DE MINIMOS CUADRADOS
# DE CADA UNO DE LOS GRUPOS
marginal <- lsmeans(model, ~ algorithm)
pairs(marginal, adjust = "sidak")

# SE IDENTIFICAN LOS GRUPOS DIFERENTES EN LOS QUE SE CLASIFICAN LOS ALGORITMOS
# EN ESTUDIO
cld <- cld(marginal, alpha = 0.05, Letters = letters, adjust = "sidak")
cld

# seed
marginal <- lsmeans(model, ~ seed)
pairs(marginal, adjust = "sidak")

# SE IDENTIFICAN LOS GRUPOS DIFERENTES EN LOS QUE SE CLASIFICAN LOS seed
# EN ESTUDIO
cld <- cld(marginal, alpha = 0.05, Letters = letters, adjust = "sidak")
cld

# time
marginal <- lsmeans(model, ~ time)
pairs(marginal, adjust = "sidak")

# SE IDENTIFICAN LOS GRUPOS DIFERENTES EN LOS QUE SE CLASIFICAN LOS time
# EN ESTUDIO
cld <- cld(marginal, alpha = 0.05, Letters = letters, adjust = "sidak")
cld

# Summarize mean result vs algorithm
sum <- Summarize(result ~ algorithm, data = data, digits = 3)

sum$se <- sum$sd / sqrt(sum$n)
sum$se <- signif(sum$se, digits = 3)
sum

sum$algorithm <- factor(sum$algorithm, levels = unique(sum$algorithm))

#GRAFICO
pd <- position_dodge(.2)

plot <- ggplot(sum, aes(x = algorithm,
                y = mean,
                color = algorithm)) +
  geom_errorbar(aes(ymin = mean - se,
                    ymax = mean + se),
                    width = 0.2,
                    size = 0.7, position = pd) +
  geom_point(shape = 15, size = 4, position = pd) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold")) +
  scale_colour_manual(values = c("black", "red", "green")) +
  ylab("result")

# Increase the font size to 20 (adjust as needed)
plot + theme(axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    legend.text = element_text(size = 20)) +
  scale_size(guide = "none")

# Summarize mean result vs seed in function of the algorithm
sum <- Summarize(result ~ seed + algorithm, data = data, digits = 3)

sum$se <- sum$sd / sqrt(sum$n)
sum$se <- signif(sum$se, digits = 3)
sum

sum$seed <- factor(sum$seed, levels = unique(sum$seed))
#sum$seed <- factor(sum$seed, levels(sum$seed)[c(7,5,2,8,3,6,1,4)])

#GRAFICO
pd <- position_dodge(.2)

plot <- ggplot(sum, aes(x = seed,
                y = mean,
                color = algorithm)) +
  geom_errorbar(aes(ymin = mean - se,
                    ymax = mean + se),
                    width = 0.2,
                    size = 0.7, position = pd) +
  geom_point(shape = 15, size = 4, position = pd) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold")) +
  scale_colour_manual(values = c("black", "red", "green")) +
  ylab("result")
# Increase the font size to 20 (adjust as needed)
plot + theme(axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    legend.text = element_text(size = 20)) +
  scale_size(guide = "none")

# Summarize mean result vs time in function of the algorithm
sum <- Summarize(result ~ time + algorithm, data = data, digits = 3)

sum$se <- sum$sd / sqrt(sum$n)
sum$se <- signif(sum$se, digits = 3)
sum

sum$time <- factor(sum$time, levels = unique(sum$time))

#GRAFICO
pd <- position_dodge(.2)

plot <- ggplot(sum, aes(x = time,
                y = mean,
                color = algorithm)) +
  geom_errorbar(aes(ymin = mean - se,
                    ymax = mean + se),
                    width = 0.2,
                    size = 0.7, position = pd) +
  geom_point(shape = 15, size = 4, position = pd) +
  theme_bw() +
  theme(axis.title = element_text(face = "bold")) +
  scale_colour_manual(values = c("black", "red", "green")) +
  ylab("result")
  
# Increase the font size to 20 (adjust as needed)
plot + theme(axis.title = element_text(size = 20),
    axis.text = element_text(size = 16),
    legend.text = element_text(size = 20)) +
    scale_size(guide = "none")

pairwise.t.test(data$result, data$algorithm, p.adjust.method = 'BH')
pairwise.t.test(data$result, data$seed, p.adjust.method = 'BH')
pairwise.t.test(data$result, data$time, p.adjust.method = 'BH')

####################################################################################################################
library(rcompanion)
t_sqrt <- sqrt(data$result)

modelt <- lm(t_sqrt ~ algorithm * seed * time, data = data)

library(car)
Anova(modelt, type = "II") # Suma de cuadrados

xt <- residuals(modelt)

library(rcompanion)
plotNormalHistogram(xt)
# Homocedasticidad
plot(fitted(modelt), residuals(modelt))

plot(modelt)

####################################################################################################################
library(rcompanion)
t_cub <- sign(data$result) * abs(data$result)^(1 / 3)

model_cube <- lm(t_cub ~ algorithm * seed * time, data = data)

library(car)
Anova(model_cube, type = "II") # Suma de cuadrados

xcube <- residuals(model_cube)

library(rcompanion)
plotNormalHistogram(xcube)

# Homocedasticidad
plot(fitted(model_cube), residuals(model_cube))

plot(model_cube)

###################################################################################################################
library(rcompanion)
t_log <- log(data$result)

model_log <- lm(t_log ~ algorithm * seed * time, data = data)

library(car)
Anova(model_log, type = "II") # Suma de cuadrados

xlog <- residuals(model_log)

library(rcompanion)
plotNormalHistogram(xlog)

# Homocedasticidad
plot(fitted(model_log), residuals(model_log))

plot(model_log)
