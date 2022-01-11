import json
import matplotlib.pyplot as plt
import random
import matplotlib
import numpy as np
import pandas as pd
import gsw
import pickle
from os.path import basename
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from holteandtalley import HolteAndTalley
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

np.random.seed(2)
def extract_identifications(fname,exlcude):
    list_of_names = {}
    names = {}
    unique_names = set()
    with open(fname) as f:
        for line in f:
            o= json.loads(line)
            print(o)
            unique_names.add(o["identifierName"])
            if o["profileName"] not in names and int(o["depth"])>0:
                names[o["profileName"]] = {}
                names[o["profileName"]]["depths"]=[]
                names[o["profileName"]]["identifierNames"]=[]
            if int(o["depth"])>0 and o["identifierName"] not in exclude:
                names[o["profileName"]]["depths"].append(o["depth"])
                names[o["profileName"]]["identifierNames"].append(o["identifierName"])
                list_of_names.setdefault(o["identifierName"],0)
                list_of_names[o["identifierName"]] = list_of_names[o["identifierName"]] +1
        print(unique_names)
        for n in names.keys():
            depths = np.asarray(names[n]["depths"])
            m = np.nanmean(depths)
            s = np.std(depths)
            depths = depths[np.logical_and(depths>m-2*s,depths<m+2*s)]
            names[n]["depths"] = depths

        return names

exclude = ["gdf", "professor dr teacher", "gdoggerino", "Garrett!", "fullrun", "Garrett Finucane", "garrett", "garrett finucane","Gdf","g doggerino", "mike", "Michael"]

names = extract_identifications("output.json",exclude)

pickle.dump(names,open("identifications.pickle","wb"))

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


def ht_features(profile):
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

def ht_features_reduced(profile):
    plt.plot(profile["densities"],profile["pressures"])
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
    return tempfactors + densfactors

def findOutliers(names):
    for p in names.values():
        depths=np.asarray(p["depths"])
        divergent = np.asarray(p["identifierNames"])[np.abs(depths-np.mean(depths))>50]
        print(divergent)

def extract_argo_and_split(fname,training_size):
    with open(fname) as f:
        profiles = json.load(f)
        training_size = 0.5
        lowlatprofiles = []
        highlatprofiles = []
        for p in profiles:
            if np.abs(p["lat"])>60:
                highlatprofiles.append(p)
            else:
                lowlatprofiles.append(p)
        chosenlow = np.random.choice(len(lowlatprofiles), size=int(len(lowlatprofiles)*training_size), replace=False)
        chosenhigh = np.random.choice(len(highlatprofiles), size=int(len(highlatprofiles)*training_size), replace=False)
        lowmask = np.zeros_like(lowlatprofiles,bool)
        lowmask[chosenlow] = True
        highmask = np.zeros_like(highlatprofiles,bool)
        highmask[chosenhigh] = True
    return lowlatprofiles+highlatprofiles,np.concatenate((lowmask,highmask))

def create_training_data(profiles,chosenprofiles,names,feature_function=ht_features_reduced):
    X = []
    y = []
    for profile in np.asarray(profiles)[chosenprofiles]:
        if profile["name"] in names.keys():
            htp = np.asarray(feature_function(profile))
            if len(names[profile["name"]]["depths"]) >0:
                X.append(np.asarray(htp))
                y.append(np.nanmean(names[profile["name"]]["depths"]))
    return X,y

def quality_index(profile,mld_depth):
    pres = np.asarray(profile["pressures"])
    dens = np.asarray(profile["densities"])
    uppervar = np.std(dens[pres<mld_depth])
    lowervar = np.std(dens[pres<mld_depth*1.5])
    return 1 - (uppervar/lowervar)


profiles,chosenprofiles = extract_argo_and_split('../MLDIdentifierTool/json_generator/profiles.json',0.25)
X,y = create_training_data(profiles,chosenprofiles,names)

y = np.asarray(y)
X = np.asarray(X)
## Random Forest
### REALLY GOOD FOR SOME REASON
regr = RandomForestRegressor(random_state=0, oob_score=True,n_jobs=5)
#Next
#regr = RandomForestRegressor(n_jobs=5,max_depth=20,random_state=0,n_estimators=1000)
regr.fit(X,y)
#feature_names = ["lon","lat","doy","tmax","mltfittemp","ttmld","dtm","tdtm","dmin","mltfitdens","dthreshold","dgradient","sgradmax","smltfit","intrusiondepth"]
#feature_names = ["lon","lat","doy"]
feature_names = ["DBM (Temperature)", "MLTFIT (Density)", "DBM (Density)"]
#dens_names = list(np.arange(0.01,0.05,0.001))
#temp_names = list(np.arange(0.01,0.3,0.01))
#feature_names += dens_names + temp_names

importances = regr.feature_importances_
print("most important feature:" , np.asarray(feature_names)[np.argmax(importances)])
forest_importances = pd.Series(importances, index=feature_names)
fig, ax = plt.subplots(figsize=(8, 6))
forest_importances.plot.bar(ax=ax)
ax.set_ylabel ("Normalized Feature Importance")
plt.title("Relative Feature Importance")
plt.savefig("feature_importance.png")
plt.close()
# A sensitive one gdfcdfs/jma/2902997/profiles/R2902997_034.nc
# gdfcdfs/aoml/2903126/profiles/R2903126_176.nc
##################
### Boyer Montegut
def error_dist_figure(profiles,chosenprofiles,mlmodel,feature_function):
    ht_error = []
    crit_error = []
    regr_error = []
    obs_std = []
    for profile in np.asarray(profiles)[~chosenprofiles]:
        if profile["name"] in names.keys():
            depths=np.asarray(names[profile["name"]]["depths"])
            if len(depths)>0:
                h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
                X = np.asarray([feature_function(profile)])
                regr_out = np.round(mlmodel.predict(X)[0])
                regr_error.append((regr_out-np.mean(depths)))
                regr_error[-1] = np.abs(regr_error[-1]/np.mean(depths))
                g = interpolate.interp1d(profile["pressures"],profile["densities"])
                ht_error.append((h.densityMLD-np.mean(depths)))
                ht_error[-1] = np.abs(ht_error[-1]/np.mean(depths))
                obs_std.append(np.std(depths)/np.mean(depths))
                try:
                    #plt.scatter(f(h.temp.TTMLD),h.temp.TTMLD,color="orange")
                    crit_error.append((h.density.DThresholdPressure-np.mean(depths)))
                    crit_error[-1] =np.abs(crit_error[-1]/np.mean(depths))
                except:
                    crit_error.append(np.nan)
                    print("out of range")
    error = np.asarray([regr_error,ht_error,crit_error,obs_std])*100
    plt.close()
    fig, ax = plt.subplots(figsize=(12,15))
    ax.set_ylabel("Percent Of Mean Visually Identified Mixed Layer")
    labels = ["RF Error","H&T (Density) Error","DBM (Density) Error","Obs. STDEV"]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.boxplot(error.T,labels=["RF Error","H&T (Density) Error","Density Thresh Error","Obs. STDEV"])
    plt.title("Comparison of Random Forest Model and Existing Mixed Layer Depth Algorithms")
    plt.savefig('errordist.png')
    plt.close()

def example_figure(profiles,names,eyed,mlmodel,feature_function):
    X = []
    y = []
    for profile in profiles:
        if eyed in profile["name"]:
            h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
            X = np.asarray([feature_function(profile)])
            regr_out = np.round(mlmodel.predict(X)[0])
            plt.plot(profile["densities"],profile["pressures"])
            plt.invert_yaxis()

#example_figure(profiles,names,"R6902637_097",regr,ht_features_reduced)

def quality_index_figure(profiles,chosenprofiles,mlmodel,feature_function):
    ht_error = []
    crit_error = []
    regr_error = []
    obs_std = []
    for profile in np.asarray(profiles)[~chosenprofiles]:
        if profile["name"] in names.keys():
            depths=np.asarray(names[profile["name"]]["depths"])
            if len(depths)>0:
                h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
                X = np.asarray([feature_function(profile)])
                regr_out = np.round(mlmodel.predict(X)[0])
                regr_error.append(quality_index(profile,regr_out))
                ht_error.append(quality_index(profile,h.densityMLD))
                try:
                    #plt.scatter(f(h.temp.TTMLD),h.temp.TTMLD,color="orange")
                    crit_error.append(quality_index(profile,h.density.DThresholdPressure))
                except:
                    crit_error.append(np.nan)
                    print("out of range")
    mask = np.logical_and(~np.isnan(crit_error),~np.isnan(ht_error))
    regr_error = np.asarray(regr_error)[mask]
    ht_error = np.asarray(ht_error)[mask]
    crit_error = np.asarray(crit_error)[mask]
    print(regr_error)
    print(ht_error)
    print(crit_error)
    error = np.asarray([regr_error,ht_error,crit_error])
    plt.close()
    fig, ax = plt.subplots(figsize=(12,15))
    ax.set_ylabel("")
    labels = ["RF Error","H&T (Density) Error","DBM (Density) Error"]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.boxplot(error.T,labels=["RF Error","H&T (Density) Error","Density Thresh Error"])
    plt.title("")
    plt.savefig('qual_index.png')
    plt.close()

quality_index_figure(profiles,chosenprofiles,regr,ht_features_reduced)
error_dist_figure(profiles,chosenprofiles,regr,ht_features_reduced)

def obs_std_error(profiles):
    with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
        profiles = json.load(f)
        obs_std=[]
        obs_std_norm=[]
        lats = []
        for profile in np.asarray(profiles)[~chosenprofiles]:
            depths = names[profile["name"]]["depths"]
            depths=np.asarray(depths)
            if len(depths)>0:
                obs_std.append(np.std(depths))
                obs_std_norm.append((np.std(depths)/np.nanmean(depths))*100)
                lats.append(profile["lat"])
        obs_std = np.asarray(obs_std)
        obs_std_norm = np.asarray(obs_std_norm)
        lats = np.asarray(lats)
        plt.close()
        fig, ((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(3,2,figsize=(14,8))
        ax1.hist(obs_std,color="black")
        ax1.axvline(x=np.nanmean(obs_std),color='red')
        ax1.axvline(x=np.nanmedian(obs_std),color='blue')
        print(np.median(obs_std))
        ax3.hist(obs_std[np.abs(lats)>50],color="black")
        ax3.axvline(x=np.nanmean(obs_std[np.abs(lats)>50]),color='red')
        ax3.axvline(x=np.nanmedian(obs_std[np.abs(lats)>50]),color='blue')
        ax5.hist(obs_std[np.abs(lats)<50],color="black")
        ax5.axvline(x=np.nanmean(obs_std[np.abs(lats)<50]),color='red')
        ax5.axvline(x=np.nanmedian(obs_std[np.abs(lats)<50]),color='blue')

        ax2.hist(obs_std_norm,color="black")
        ax2.axvline(x=np.nanmean(obs_std_norm),color='red')
        ax2.axvline(x=np.nanmedian(obs_std_norm),color='blue')

        ax4.hist(obs_std_norm[np.abs(lats)>50],color="black")
        ax4.axvline(x=np.nanmean(obs_std_norm[np.abs(lats)>50]),color='red')
        ax4.axvline(x=np.nanmedian(obs_std_norm[np.abs(lats)>50]),color='blue')

        ax6.hist(obs_std_norm[np.abs(lats)<50],color="black")
        ax6.axvline(x=np.nanmean(obs_std_norm[np.abs(lats)<50]),color='red')
        ax6.axvline(x=np.nanmedian(obs_std_norm[np.abs(lats)<50]),color='blue')
        ax6.set_xlabel("Standard Deviation As Perecentage of Mean")
        ax5.set_xlabel("Standard Deviation in Dbar")
        ax1.set_xlim(0,100)
        ax3.set_xlim(0,100)
        ax5.set_xlim(0,100)
        ax2.set_xlim(0,200)
        ax4.set_xlim(0,200)
        ax6.set_xlim(0,200)
        #ax2.hist(obs_std_norm[np.abs(lats)>50],color="blue")
        #ax2.hist(obs_std_norm[np.abs(lats)<50],color="red")
        plt.savefig('observational_error.png')
        plt.close()
obs_std_error(profiles)
if False:
    with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
        profiles = json.load(f)
        ht_error = []
        crit_error = []
        regr_error = []
        obs_error=[]
        lats= []
        means=[]
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
                    means.append(np.nanmean(depths))
        plt.close()
        error = np.asarray([regr_error,ht_error,obs_error,crit_error])
        plt.boxplot(error.T,labels=["RF","HT","Observed STDEV","Density Threshold"])
        plt.savefig("error_bias.png")
        plt.close()
        plt.scatter(lats,regr_error,color="blue")
        plt.scatter(lats,ht_error,color="orange")
        plt.scatter(lats,crit_error,color="red")
        plt.scatter(lats,obs_error,color="black")
        plt.savefig("lat_error.png")
        plt.close()
        plt.scatter(means,obs_error,color="black")
        plt.savefig("errorvssize.png")

if False:
    with open('../MLDIdentifierTool/json_generator/profiles.json') as f:
        profiles = json.load(f)
        error = []
        obs_std = []
        lats = []
        for profile in np.asarray(profiles):
            if profile["name"] in names.keys():
                depths = names[profile["name"]]["depths"]
                depths=np.asarray(depths)
                if len(depths)>0:
                    lats.append(profile["lat"])
                    h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
                    densfactors = [h.density.DMinPressure, h.density.MLTFITDensityPressure, h.density.DThresholdPressure, h.density.DGradientThresholdPressure,h.densityMLD] 
                    salfactors = [h.salinity.SGradientMaxPressure,h.salinity.MLTFITSalinityPressure,h.salinity.intrusionDepthPressure]
                    tempfactors =[ h.temp.TMaxPressure, h.temp.MLTFITPressure, h.temp.TTMLDPressure, h.temp.DTMPressure, h.temp.TDTMPressure,h.tempMLD]
                    htfactors = np.asarray(densfactors + salfactors + tempfactors)
                    obs_std.append(np.std(depths)/np.nanmean(depths))
                    error.append((htfactors-np.nanmean(depths))/np.nanmean(depths))

        names = ["dminpressure","mltfitdens","dthreshold","dgradientthreshold","ht-dens","sgradientmax","mltfitsalinity","intrusion","tmax","mltfit","tempthreshold","dtm","tdtmp","ht-temp"]
        lats = np.asarray(lats)
        error = np.asarray(error)
        error = np.sqrt(error**2)
        lowlatitudeerror = np.asarray(error)[lats<50,:]
        highlatitudeerror = np.asarray(error)[lats>=50,:]
        lowlatitudeerror[lowlatitudeerror<0.0001]=0.0001
        highlatitudeerror[highlatitudeerror<0.0001]=0.0001
        print(lowlatitudeerror.shape)
        print(highlatitudeerror.shape)
        plt.close()
        print(np.mean(obs_std))
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 15))
        ax1.boxplot(np.log10(lowlatitudeerror),labels=names)
        ax2.boxplot(np.log10(highlatitudeerror),labels=names)
        #for tick in ax1.get_xticklabels():
            #tick.set_rotation(45)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_xticklabels(names, rotation=45, ha='right')
        #for l in range(error.T.shape[0]):
            #print(names[l],int(np.nanmean(np.sqrt(error.T[l]**2))),int(np.std(np.sqrt(error.T[l]**2))))
        plt.savefig("ht_error.png")

def sensitivity_figure(profiles,names):
    profiles = json.load(f)
    ht_error = []
    regr_error = []
    thresh_error = []
    test_profiles = []
    for profile in np.asarray(profiles)[~chosenprofiles]:
        test_profiles.append(profile)
    plt.close()
    n_iter = 25
    for profile in test_profiles:
        orig_pressures = profile["pressures"]
        orig_salinities = profile["salinities"]
        orig_temperatures = profile["temperatures"]
        orig_densities = profile["densities"]
        ht_results = []
        thresh_results = []
        regr_results = []
        for l in range(n_iter):
            depths = names[profile["name"]]["depths"]
            profile["temperatures"] = np.asarray(orig_temperatures) + np.random.normal(0,0.002*0.5,size=len(orig_temperatures))
            profile["salinities"] = np.asarray(orig_salinities) + np.random.normal(0,0.01*0.5,size=len(orig_salinities))
            profile["densities"] = gsw.sigma0(profile["salinities"],profile["temperatures"])
            h = HolteAndTalley(profile["pressures"],profile["temperatures"],profile["salinities"],profile["densities"])
            X = np.asarray([ht_features_reduced(profile)])
            regr_out = np.round(regr.predict(X)[0])
            ht_results.append(h.densityMLD)
            thresh_results.append(h.density.DThresholdPressure)
            regr_results.append(regr_out)
        ht_error.append(np.std(ht_results))
        regr_error.append(np.std(regr_results))
        thresh_error.append(np.std(thresh_results))
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(10,10))
    ax1.hist(ht_error,range=(0,50),bins=20,color="red",label="Holte and Talley",alpha=0.3)
    ax1.set_ylim(0,60)
    ax2.set_ylim(0,60)
    ax3.set_ylim(0,60)
    ax2.hist(regr_error,range=(0,50),bins=20,color="blue",label = "Random Forest Method",alpha=0.3)
    ax3.hist(thresh_error,range=(0,50),bins=20,color="green", label = "Density Threshold Method",alpha=0.3)
    ax.legend()

sensitivity_figure(profiles,names)
