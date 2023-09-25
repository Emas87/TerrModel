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

data1 <- read.table("../code/experiments1.txt", header = TRUE)
data2 <- read.table("../code/experiments2.txt", header = TRUE)
data3 <- read.table("../code/experiments3.txt", header = TRUE)
data4 <- read.table("../code/experiments4.txt", header = TRUE)
data5 <- read.table("../code/experiments5.txt", header = TRUE)
data6 <- read.table("../code/experiments6.txt", header = TRUE)
data7 <- read.table("../code/experiments7.txt", header = TRUE)
data8 <- read.table("../code/experiments8.txt", header = TRUE)
data9 <- read.table("../code/experiments9.txt", header = TRUE)
data10 <- read.table("../code/experiments10.txt", header = TRUE)

# SE GENERA LA TABLA CON LOS DATOS Y SE ELIMINAN LOS DATOS CARGADOS
# DESDE EL ARCHIVO
data <- rbind(data1, data2, data3, data4, data5, data6, data7, data8, data9, data10) # nolint


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
leveneTest(t_sqrt ~ algorithm * seed * time, data = data)

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
leveneTest(t_cub ~ algorithm * seed * time, data = data)

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
leveneTest(t_log ~ algorithm * seed * time, data = data)

library(car)
Anova(model_log, type = "II") # Suma de cuadrados

xlog <- residuals(model_log)

library(rcompanion)
plotNormalHistogram(xlog)

# Homocedasticidad
plot(fitted(model_log), residuals(model_log))

plot(model_log)
