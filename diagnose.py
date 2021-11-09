import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from os.path import basename
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from holteandtalley import HolteAndTalley
matplotlib.use('pdf')


data = []
names = {}

exclude = ["gdf", "professor dr teacher", "gdoggerino", "Garrett!", "fullrun", "Garrett Finucane", "garrett", "garrett finucane","Gdf","g doggerino"]

count = 0
with open('output.json') as f:
    for line in f:
        o= json.loads(line)
        if o["profileName"] not in names and int(o["depth"])>0:
            names[o["profileName"]] = {}
            names[o["profileName"]]["depths"]=[]
            names[o["profileName"]]["identifierNames"]=[]
        if int(o["depth"])>0 and o["identifierName"] not in exclude:
            count+=1
            names[o["profileName"]]["depths"].append(o["depth"])
            names[o["profileName"]]["identifierNames"].append(o["identifierName"])
number=[]
stdevs=[]
mean = []
for l in names.keys():
    d  = np.asarray(names[l]["depths"])
    if len(d)>=2:
        number.append(len(d))
        mean.append(np.mean(d))
        stdevs.append(np.std(d))

with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
    profiles = json.load(f)
    doy = []
    lat = []
    lon = []
    densities = [] 
    y = [ ]
    chosenprofiles = np.random.choice(len(profiles), size=int(len(profiles)*0.25), replace=False)
    mask = np.zeros_like(profiles,bool)
    mask[chosenprofiles] = True
    chosenprofiles = mask
    for profile in np.asarray(profiles)[chosenprofiles]:
        if profile["name"] in names.keys():
            plt.plot(profile["densities"],profile["pressures"])
            depths = names[profile["name"]]["depths"]
            denschoose = []
            for d in depths:
                if d >0:
                    denschoose.append(profile["densities"][profile["pressures"].index(d)])
                else:
                    denschoose.append(np.nan)
            f = interpolate.interp1d(profile["pressures"],profile["densities"])
            xnew = np.arange(20, 150, 1)
            ynew = f(xnew)
            featuredens = ynew
            y= y+depths
            lon=lon+[profile["lon"]]*len(denschoose)
            lat=lat + [profile["lat"]]*len(denschoose)
            for l in range(len(denschoose)):
                densities.append(list(featuredens)+list(np.diff(featuredens)))
            date = profile["date"]
            doy += [int(date[5:7])*30 + int(date[8:10])] * len(denschoose)

            #plt.scatter(denschoose,depths)
            #plt.savefig('./pics/{}.png'.format(basename(profile["name"])))
            #plt.close()

doy =np.asarray(doy).reshape((len(doy),1))
lon =np.asarray(lon).reshape((len(doy),1))
lat =np.asarray(lat).reshape((len(doy),1))
densities =np.asarray(densities)
y = np.asarray(y)
X = np.hstack((doy,lon,lat,densities))
## DECISION TREE
dec_tree = DecisionTreeRegressor()
dec_tree.fit(X,y)
#y_pred_dec_tree=dec_tree.predict(X)
#print("RMSE: ",np.sqrt(np.mean((y_test-y_pred_dec_tree)**2)))
#print(X_test[:,0].T.shape,y_pred_dec_tree.shape)
#plt.scatter(X_test[:,0].T,y_pred_dec_tree,label = "Decision Tree")
##################
## Random Forest
### REALLY GOOD FOR SOME REASON
#regr = RandomForestRegressor(random_state=0, oob_score=True, max_depth=7)
#Next
regr = RandomForestRegressor(random_state=0, oob_score=True)
regr.fit(X,y)
#y_pred_random=regr.predict(X_test)
#print("oob score", regr.oob_score_)
#print("RMSE: ",np.sqrt(np.mean((y_test-y_pred_random)**2)))
#print("stdev: ",np.std(np.abs((y_test-y_pred_random))))
#plt.scatter(X_test[:,0],y_pred_random,label = "Random Forest")
#plt.scatter(X_test[:,0],y_test,label="True")
#plt.savefig("comparison.png")
#plt.close()
##################
### Boyer Montegut
dec_error= []
ht_error = []
crit_error = []
obs_std = []
print(len(np.asarray(profiles)[~chosenprofiles]))
with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
    profiles = json.load(f)
    for profile in np.asarray(profiles)[~chosenprofiles]:
        if profile["name"] in names.keys():
            depths = names[profile["name"]]["depths"]
            denschoose = []
            for d in depths:
                if d >0:
                    denschoose.append(profile["densities"][profile["pressures"].index(d)])
                else:
                    denschoose.append(np.nan)
            f = interpolate.interp1d(profile["pressures"],profile["densities"])
            h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
            xnew = np.arange(20, 150, 1)
            ynew = f(xnew)
            date = profile["date"]
            doy = int(date[5:7])*30 + int(date[8:10])
            X = np.asarray([[doy,profile["lon"],profile["lat"]]+list(ynew)+list(np.diff(ynew))])

            X = np.asarray(X)
            dec_out = dec_tree.predict(X)[0]
            regr_out = regr.predict(X)[0]
            mask = np.asarray(profile["pressures"])<200
            g = interpolate.interp1d(profile["pressures"],profile["temperatures"])
            #plt.plot(np.asarray(profile["temperatures"])[mask],np.asarray(profile["pressures"])[mask],color="black")
            #plt.scatter(denschoose,depths,color="red")
            #plt.scatter(f(dec_out),dec_out,color="blue")
            dec_error.append((regr_out-np.mean(depths))**2)
            #plt.scatter(f(regr_out),regr_out,color="green")
            ht_error.append((h.tempMLD-np.mean(depths))**2)
            #plt.scatter(f(h.tempMLD),h.tempMLD,color="black")
            obs_std.append(np.std(depths))
            try:
                #plt.scatter(f(h.temp.TTMLD),h.temp.TTMLD,color="orange")
                crit_error.append((h.temp.TTMLD-np.mean(depths))**2)
            except:
                crit_error.append(np.nan)
                print("out of range")

            #plt.savefig('./pics/{}.png'.format(basename(profile["name"])))
            plt.close()
dec_error = np.sqrt(np.asarray(dec_error))
ht_error = np.sqrt(np.asarray(ht_error))
crit_error = np.sqrt(np.asarray(crit_error))
#plt.hist(dec_error,label="dec",alpha=0.5,color="red")
#plt.hist(ht_error,label="ht",alpha=0.5,color="yellow")
#plt.hist(crit_error,label="crit",alpha=0.5,color="blue")
#plt.hist(range(len(obs_std)),obs_std,c="blue")
plt.scatter(obs_std,dec_error,c="red",label="dec")
plt.scatter(obs_std,ht_error,c="blue",label="ht")
plt.scatter(obs_std,crit_error,c="yellow",label="crit")
plt.plot(obs_std,obs_std,c="orange",label="obs")
plt.legend()
print("dec mean",np.nanmean(dec_error), " dec max ", np.nanmax(dec_error))
print("ht mean",np.nanmean(ht_error), " dht max ", np.nanmax(ht_error))
print("crit mean",np.nanmean(crit_error), " crit max ", np.nanmax(crit_error))
print("obs stdev mean",np.nanmean(obs_std), " dec max ", np.nanmax(obs_std))
plt.savefig('errorcomp.png')
plt.close()


