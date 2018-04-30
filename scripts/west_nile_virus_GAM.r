setwd('~/Documents/Kaggle/nile_virus/scripts/')
options(scipen = 999) # no scientific number

library(lubridate)
library(AUC) 
library(plyr)
library(gbm)
library(zoo)
library(mgcv) # GAM model

weather = read.csv('../input/weather.csv')
dat = read.csv('../input/train.csv')
dat$Date = ymd(dat$Date)
dat$WnvPresent = as.factor(dat$WnvPresent)
weather$Date = ymd(weather$Date)
dat$Station = 1

### Processing data

temp = merge(dat, weather)
temp$month = month(temp$Date)
temp$week = week(temp$Date)
temp$year = year(temp$Date) ## Seems like recent year has higher Wnv rate
temp$dayofyear = as.numeric(yday(temp$Date))
temp$PrecipTotal = as.numeric(as.character(temp$PrecipTotal))
temp$PrecipTotal[is.na(temp$PrecipTotal)] = 0
temp$WetBulb = as.numeric(as.character(temp$WetBulb))
temp$WetBulb[is.na(temp$WetBulb)] = 0
temp$StnPressure = as.numeric(as.character(temp$StnPressure))
temp$StnPressure[is.na(temp$StnPressure)] = 0
temp$AvgSpeed = as.numeric(as.character(temp$AvgSpeed))
temp$AvgSpeed[is.na(temp$AvgSpeed)] = 0
train = temp[, c('week', 'month','Species' ,'Latitude',
                'Longitude', 'Tmax', 'PrecipTotal' , "NumMosquitos")]
train[train$Species == 'CULEX ERRATICUS' | train$Species == 'CULEX TARSALIS', 'Species'] = 'CULEX TERRITANS'

y = as.factor(temp$WnvPresent)

td = temp$Date # Date variable for cross validation

### GAM model 
for(yr in c(2007, 2009, 2011, 2013)){
  #Split data
  holdout = train[year(td) ==yr, ]
  holdout.y = y[year(td) ==yr]
  train.cv = train[year(td) != yr,]
  train.y = y[year(td) != yr]
  d2 = cbind(train.cv, train.y)
  fitted = mgcv::gam(train.y ~ ti(Latitude, Longitude) + s(PrecipTotal) + 
                       s(Tmax) + s(NumMosquitos) + s(week) + s(week) + Species , data = d2,family = binomial(),
                     select = TRUE)
  
  ypred = predict(fitted, newdata = holdout, type = 'response')
  res = auc(roc(ypred, holdout.y))
  print(paste("year", yr, res))
}
