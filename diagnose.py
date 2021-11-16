import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from os.path import basename
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from holteandtalley import HolteAndTalley
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.neural_network import MLPRegressor
matplotlib.use('pdf')


data = []
names = {}

exclude = ["gdf", "professor dr teacher", "gdoggerino", "Garrett!", "fullrun", "Garrett Finucane", "garrett", "garrett finucane","Gdf","g doggerino", "mike", "Michael"]
list_of_names = {}
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
            list_of_names.setdefault(o["identifierName"],0)
            list_of_names[o["identifierName"]] = list_of_names[o["identifierName"]] +1
print(list_of_names)
number=[]
stdevs=[]
mean = []
for l in names.keys():
    d  = np.asarray(names[l]["depths"])
    if len(d)>=2:
        number.append(len(d))
        mean.append(np.mean(d))
        stdevs.append(np.std(d))
plt.hist(number)
plt.savefig("data dist.png")
plt.close()

def vals_and_grad(profile,names):
    plt.plot(profile["densities"],profile["pressures"])
    depths = names[profile["name"]]["depths"]
    if len(depths)>0:
        #depths = list([int(np.nanmean(depths))])
        d = interpolate.interp1d(profile["pressures"],profile["densities"])
        s = interpolate.interp1d(profile["pressures"],profile["salinities"])
        t = interpolate.interp1d(profile["pressures"],profile["temperatures"])
        xnew = np.arange(20, 150, 1)
        densnew = d(xnew)
        salnew = s(xnew)
        tempnew = t(xnew)
        date = profile["date"]
        lon = profile["lon"]
        lat = profile["lat"]
        doy = int(date[5:7])*30 + int(date[8:10])
        hydrodata = list(densnew)+list(np.diff(densnew)) + list(salnew)+list(np.diff(salnew))+list(tempnew)+list(np.diff(tempnew))
        return [lon,lat,doy] + hydrodata

def density_diff(profile,names):
    plt.plot(profile["densities"],profile["pressures"])
    depths = names[profile["name"]]["depths"]
    if len(depths)>0:
        #depths = list([int(np.nanmean(depths))])
        d = interpolate.interp1d(profile["pressures"],profile["densities"])
        s = interpolate.interp1d(profile["pressures"],profile["salinities"])
        t = interpolate.interp1d(profile["pressures"],profile["temperatures"])
        xnew = np.arange(20, 150, 1)
        densnew = d(xnew)
        salnew = s(xnew)
        tempnew = t(xnew)
        hydrodata = list(densnew-densnew[0])
        return hydrodata

def find_threshold(quant,pressures,thresh):
    for index in range(0,len(pressures)):
        if abs(quant[index] - quant[0]) > thresh:
            return pressures[index]
    return 0

def thresholds(profile,name):
    pressures = profile["pressures"]
    temperatures = profile["temperatures"]
    salinities = profile["salinities"]
    densities = profile["densities"]
    date = profile["date"]
    lon = profile["lon"]
    lat = profile["lat"]
    doy = int(date[5:7])*30 + int(date[8:10])
    startindex = np.argmin((np.asarray(pressures)-10.0)**2)
    salinities = salinities[startindex:]
    pressures = pressures[startindex:]
    densities = densities[startindex:]
    temperatures = temperatures[startindex:]
    densitythresholds = np.arange(0.01,0.05,0.001)
    tempthresholds = np.arange(0.01,0.3,0.01)
    features = []
    for l in list(densitythresholds):
        features.append(find_threshold(densities,pressures,l))
    for l in list(tempthresholds):
        features.append(find_threshold(temperatures,pressures,l))
    return [lon,lat,doy] + features


def ht_features(profile,names):
    plt.plot(profile["densities"],profile["pressures"])
    depths = names[profile["name"]]["depths"]
    if len(depths)>0:
        #depths = list([int(np.nanmean(depths))])

        d = interpolate.interp1d(profile["pressures"],profile["densities"])
        xnew = np.arange(20, 150, 1)
        densnew = d(xnew)

        date = profile["date"]
        lon = profile["lon"]
        lat = profile["lat"]
        doy = int(date[5:7])*30 + int(date[8:10])
        h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
        densfactors = [h.density.DMinPressure, h.density.MLTFITDensityPressure, h.density.DThresholdPressure, h.density.DGradientThresholdPressure ] 
        salfactors = [h.salinity.SGradientMaxPressure,h.salinity.MLTFITSalinityPressure,h.salinity.intrusionDepthPressure]
        tempfactors =[ h.temp.TMaxPressure, h.temp.MLTFITPressure, h.temp.TTMLDPressure, h.temp.DTMPressure, h.temp.TDTMPressure]
        return [lon,lat,doy] + tempfactors + densfactors + salfactors 

def ht_features_reduced(profile,names):
    plt.plot(profile["densities"],profile["pressures"])
    depths = names[profile["name"]]["depths"]
    if len(depths)>0:
        #depths = list([int(np.nanmean(depths))])

        d = interpolate.interp1d(profile["pressures"],profile["densities"])
        xnew = np.arange(20, 150, 1)
        densnew = d(xnew)

        date = profile["date"]
        lon = profile["lon"]
        lat = profile["lat"]
        doy = int(date[5:7])*30 + int(date[8:10])
        h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
        densfactors = [ h.density.MLTFITDensityPressure, h.density.DThresholdPressure] 
        tempfactors =[h.temp.TTMLDPressure]
        return [np.abs(lat)*5] + tempfactors + densfactors


with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
    profiles = json.load(f)
    doy = []
    lat = []
    lon = []
    densities = [] 
    y = [ ]
    training_size = 0.5
    chosenprofiles = np.random.choice(len(profiles), size=int(len(profiles)*training_size), replace=False)
    mask = np.zeros_like(profiles,bool)
    mask[chosenprofiles] = True
    lats = []
    for profile in range(len(np.asarray(profiles))):
        if np.abs(profiles[profile]["lat"])>60:
            lats.append(profile)
            mask[profile]=False
    highlat = np.random.choice(len(lats), size=int(len(lats)*training_size), replace=False)
    mask[highlat]=True
    chosenprofiles = mask

    X = []
    for profile in np.asarray(profiles)[chosenprofiles]:
        if profile["name"] in names.keys():
            htp = np.asarray(ht_features_reduced(profile,names))
            #for l in names[profile["name"]]["depths"]:
                #X.append(htp)
                #y.append(l)
            if len(names[profile["name"]]["depths"]) >0:
                X.append(np.asarray(htp))
                y.append(np.nanmean(names[profile["name"]]["depths"]))

print(len(y),len(X))
y = np.asarray(y)
X = np.asarray(X)
## DECISION TREE
dec_tree = DecisionTreeRegressor()
dec_tree.fit(X,y)
## Random Forest
### REALLY GOOD FOR SOME REASON
#regr = RandomForestRegressor(random_state=0, oob_score=True)
#Next
regr = RandomForestRegressor(n_jobs=5,max_depth=5)
#regr = MLPRegressor(max_iter=100000,hidden_layer_sizes=(50,20,5))
#regr = GradientBoostingRegressor()
regr.fit(X,y)
#feature_names = ["lon","lat","doy","tmax","mltfittemp","ttmld","dtm","tdtm","dmin","mltfitdens","dthreshold","dgradient","sgradmax","smltfit","intrusiondepth"]
#feature_names = ["lon","lat","doy"]
feature_names = []
#dens_names = list(np.arange(0.01,0.05,0.001))
#temp_names = list(np.arange(0.01,0.3,0.01))
#feature_names += dens_names + temp_names
f_names = [f"feature {i}" for i in range(X.shape[1]-len(feature_names))]
feature_names += f_names

importances = dec_tree.feature_importances_
print("most important feature:" , np.asarray(feature_names)[np.argmax(importances)])
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots(figsize=(8, 6))
forest_importances.plot.bar(ax=ax)
plt.savefig("feaure_importance.png")
plt.close()
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
if True:
    dec_error= [[],[]]
    ht_error = [[],[]]
    crit_error = [[],[]]
    obs_std = [[],[]]
    print(len(np.asarray(profiles)[~chosenprofiles]))
    with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
        profiles = json.load(f)
        for profile in np.asarray(profiles)[~chosenprofiles]:
            if profile["name"] in names.keys():
                depths = names[profile["name"]]["depths"]
                depths=np.asarray(depths)
                if len(depths)>0:
                    h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
                    X = np.asarray([ht_features_reduced(profile,names)])
                    dec_out = dec_tree.predict(X)[0]
                    regr_out = np.round(regr.predict(X)[0])
                    mask = np.asarray(profile["pressures"])<200
                    g = interpolate.interp1d(profile["pressures"],profile["densities"])
                    denschoose = []
                    maxpres = np.nanmax(list(depths) + [regr_out,h.densityMLD,h.density.DThresholdPressure,np.mean(depths)])
                    for d in depths:
                        denschoose.append(g(d))
                    try:
                        plt.close()
                        plt.plot(np.asarray(profile["densities"])[mask],np.asarray(profile["pressures"])[mask],color="black")
                        plt.scatter(denschoose,depths,color="red",label="chosen",marker="+")
                        plt.axhline(regr_out)
                        plt.scatter(g(h.densityMLD),h.densityMLD,color="green", label="HT",marker="s")
                        plt.scatter(g(h.density.DThresholdPressure),h.density.DThresholdPressure,color="orange",marker="_",label="Dthresh")
                        plt.scatter(g(np.mean(depths)),np.mean(depths),color="black",marker="o",label="chosen mean")
                        plt.gca().invert_yaxis()
                        plt.ylim(0,maxpres)
                        plt.legend()
                        plt.savefig('./pics/{}-{}.png'.format(basename(profile["name"]),abs(int(h.densityMLD-np.mean(depths)))))
                        plt.close()
                    except:
                        print("zoped")

                    dec_error[0].append(regr_out-np.mean(depths))
                    dec_error[0][-1] = np.abs(dec_error[0][-1]/np.mean(depths))
                    dec_error[1] += list(np.abs(regr_out-depths)/np.mean(depths))
                    ht_error[0].append((h.densityMLD-np.mean(depths)))
                    ht_error[0][-1] = np.abs(ht_error[0][-1]/np.mean(depths))
                    ht_error[1] += list(np.abs(h.densityMLD-depths)/np.mean(depths))
                    obs_std[0].append(np.std(depths)/np.mean(depths))
                    obs_std[1]+= list(np.abs(np.mean(depths)-depths)/np.mean(depths))
                    try:
                        #plt.scatter(f(h.temp.TTMLD),h.temp.TTMLD,color="orange")
                        crit_error[0].append((h.density.DThresholdPressure-np.mean(depths)))
                        crit_error[0][-1] =np.abs(crit_error[0][-1]/np.mean(depths))
                        crit_error[1] += list(np.abs(h.density.DThresholdPressure-depths)/np.mean(depths))
                    except:
                        crit_error.append(np.nan)
                        print("out of range")

    #dec_error[0] = np.sqrt(np.asarray(dec_error[0]))
    #ht_error[0] = np.sqrt(np.asarray(ht_error[0]))
    #crit_error[0] = np.sqrt(np.asarray(crit_error[0]))
    hta = np.asarray(ht_error[0])
    deca = np.asarray(dec_error[0])
    crit = np.asarray(crit_error[0])
    obs_std = np.asarray(obs_std[0])
    obs_std[obs_std < 0.0001] = 0.0001
    hta[hta<0.0001] = 0.0001
    crit[crit<0.0001] = 0.0001
    deca[deca<0.0001] = 0.0001
    print(crit)
    #plt.hist(dec_error,label="dec",alpha=0.5,color="red")
    #plt.hist(ht_error,label="ht",alpha=0.5,color="yellow")
    #plt.hist(crit_error,label="crit",alpha=0.5,color="blue")
    #plt.hist(range(len(obs_std)),obs_std,c="blue")
    #plt.scatter(obs_std,dec_error,c="red",label="dec")
    #plt.scatter(obs_std,ht_error,c="blue",label="ht")
    #plt.scatter(obs_std,crit_error,c="yellow",label="crit")
    #plt.plot(obs_std,obs_std,c="orange",label="obs")
    #plt.legend()
    ## mean error from mean
    #print(" mean error from mean")
    #print("dec mean",np.nanmean(dec_error[0]), " dec max ", np.nanmax(dec_error[0]))
    #print("ht mean",np.nanmean(ht_error[0]), " dht max ", np.nanmax(ht_error[0]))
    #print("crit mean",np.nanmean(crit_error[0]), " crit max ", np.nanmax(crit_error[0]))
    #print("obs stdev mean",np.nanmean(obs_std), " dec max ", np.nanmax(obs_std))
    ### mean error
    #print(" mean error")
    #print("dec mean",np.nanmean(dec_error[1]), " dec max ", np.nanmax(dec_error[1]))
    #print("ht mean",np.nanmean(ht_error[1]), " dht max ", np.nanmax(ht_error[1]))
    #print("crit mean",np.nanmean(crit_error[1]), " crit max ", np.nanmax(crit_error[1]))
    #print("obs stdev mean",np.nanmean(obs_std), " dec max ", np.nanmax(obs_std))
    #plt.savefig('errorcomp.png')
    error = np.log10(np.asarray([deca,hta,obs_std,crit]))
    plt.boxplot(error.T,labels=["RF","HT","Observed STDEV","Density Threshold"])
    plt.savefig('errordist.png')
    plt.close()

if True:
    with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
        profiles = json.load(f)
        ht_error = []
        crit_error = []
        regr_error = []
        obs_error=[]
        lats= []
        for profile in np.asarray(profiles)[~chosenprofiles]:
            if profile["name"] in names.keys():
                depths = names[profile["name"]]["depths"]
                depths=np.asarray(depths)
                if len(depths)>0:
                    h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
                    X = np.asarray([ht_features_reduced(profile,names)])
                    dec_out = dec_tree.predict(X)[0]
                    regr_out = regr.predict(X)[0]
                    lats.append(profile["lat"])
                    crit_error.append(h.density.DThresholdPressure-np.mean(depths))
                    ht_error.append(h.densityMLD-np.mean(depths))
                    regr_error.append(regr_out-np.mean(depths))
                    obs_error.append(np.std(depths))
        plt.close()
        error = np.log10(np.asarray([regr_error,ht_error,obs_error,crit_error]))
        plt.boxplot(error.T,labels=["RF","HT","Observed STDEV","Density Threshold"])
        plt.savefig('errordist.png')
        plt.xlim(-100,100)
        plt.savefig("error_bias.png")
        plt.close()
        plt.scatter(lats,regr_error,color="blue")
        plt.scatter(lats,ht_error,color="orange")
        plt.scatter(lats,crit_error,color="red")
        plt.scatter(lats,obs_error,color="black")
        plt.savefig("lat_error.png")
        plt.close()

if False:
    with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
        profiles = json.load(f)
        error = []
        obs_std = []
        lats = []
        for profile in np.asarray(profiles):
            if profile["name"] in names.keys() and np.abs(profile["lat"])<50:
                depths = names[profile["name"]]["depths"]
                depths=np.asarray(depths)
                if len(depths)>0:
                    h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
                    densfactors = [h.density.DMinPressure, h.density.MLTFITDensityPressure, h.density.DThresholdPressure, h.density.DGradientThresholdPressure ] 
                    salfactors = [h.salinity.SGradientMaxPressure,h.salinity.MLTFITSalinityPressure,h.salinity.intrusionDepthPressure]
                    tempfactors =[ h.temp.TMaxPressure, h.temp.MLTFITPressure, h.temp.TTMLDPressure, h.temp.DTMPressure, h.temp.TDTMPressure]
                    htfactors = np.asarray(densfactors + salfactors + tempfactors)
                    obs_std.append(np.std(depths)/np.nanmean(depths))
                    error.append((htfactors-np.nanmean(depths))/np.nanmean(depths))

        names = ["dminpressure","mltfitdens","dthreshold","dgradientthreshold","sgradientmax","mltfitsalinity","intrusion","tmax","mltfit","tempthreshold","dtm","tdtmp"]
        error = np.asarray(error)
        plt.close()
        print(np.mean(obs_std))
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.boxplot(np.log10(np.sqrt(error**2)),labels=names)
        #for l in range(error.T.shape[0]):
            #print(names[l],int(np.nanmean(np.sqrt(error.T[l]**2))),int(np.std(np.sqrt(error.T[l]**2))))
        plt.savefig("ht_error.png")
