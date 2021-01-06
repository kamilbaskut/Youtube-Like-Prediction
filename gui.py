import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np
import time
import threading


from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing


from selenium import webdriver 
from selenium.webdriver.common.by import By 
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.support import expected_conditions as EC

import tkinter
from tkinter import *
from tkinter import ttk

class GUI():
    def __init__(self,gui):
        self.gui = gui
        self.parameters = {'dakika' : 1,
              'saat' : 60,
              'gün' : 24*60,
              'hafta' :7*24*60,
              'ay' : 30*24*60,
              'yıl' :  365*30*24*60}

        style = ttk.Style() 
        
        return
    
    def screen(self):
        self.welcomeFrame = Frame(self.gui)
        self.welcomeFrame.config(bg = '#282828')
        
        self.welcomeText = Label(self.welcomeFrame,text = "YouTube Like Prediction")
        self.welcomeText.config(bg = '#282828' , font = ('helvetica',50) ,fg = 'white')
        
        self.welcomeText.pack()
        
        self.skipButton = Button(self.welcomeFrame,text = "Skip",command = self.mainScreen,width = 50)
        self.skipButton.config(bg = 'gray40' , font = ('helvetica bold',12),fg = 'white')
        self.skipButton.pack(expand = False ,fill = None)
        
        self.welcomeFrame.pack(expand = True)
        
    def mainScreen(self):
        self.welcomeFrame.pack_forget()
        
        self.welcomeText = Label(self.gui,text = "YouTube Like Prediction")
        self.welcomeText.config(bg = '#282828' , font = ('helvetica',50) ,fg = 'white')
        self.welcomeText.pack()
        
        self.main = Frame(self.gui)
        self.main.config(bg = '#282828')
        
        self.linkText = StringVar()
        self.getLinkText = Entry(self.main, font = ("helvetica",10), textvariable = self.linkText, width = 120)
        self.getLinkText.config(bg = 'gray40' , font = ('helvetica bold',12),fg = 'white')
        self.getLinkText.insert(0, 'Please paste your YouTube video link..')
        self.getLinkText.bind("<FocusIn>", lambda args: self.getLinkText.delete('0', 'end'))
        
        self.getLinkText.grid(row = 1,column = 1)


        self.submit =  Button(self.main, text = "Submit",command = self.browserThread, width = 20)
        self.submit.config(bg = 'gray40' , font = ('helvetica',10),fg = 'white')
        
        self.submit.grid(row = 1,column = 2)
        
        self.progress = ttk.Progressbar(self.main, orient = HORIZONTAL, 
                               length = 100, mode = 'determinate') 
                
        self.tables = Frame(self.gui,height = 200)

        self.main.pack(fill = 'both')
        
    def browserThread(self):
        self.t = threading.Thread(target = self.process)
        self.t.start()
    
    def process(self):
        self.progress.grid(row = 2,column = 1, sticky = "ew")
        self.tables.pack_forget()
        for widget in self.tables.winfo_children():
            widget.destroy()
        self.progress['value'] = 0
        self.submit.grid_forget()
        
        retValue = self.getDataFromURL()
        if retValue:
            numberOfViews,numberOfLikes,numberOfDisLikes,numberOfComments,dates_2,dates,titles,links = retValue
            # Verilerden dataframe oluşturulur.
            data = {'numberOfViews':numberOfViews,'dates':dates, 'numberOfDisLikes': numberOfDisLikes,\
                    'numberOfComments' :numberOfComments,'links':links,\
                    'numberOfLikes' : numberOfLikes }
                
            self.df = pd.DataFrame(data)
            df = self.df.copy()
            
            trainTest = ["train"]*len(links)
            trainTest[-1] = "test"
            self.df.insert(0, 'train/test', trainTest)
            self.df.insert(1, 'title', titles)
            self.df.drop('dates',axis = 1, inplace = True) 
            self.df.insert(2,'dates',dates_2)
            
            self.progress['value']+=5
            # En basta yerlestirdigimiz test linkinin özellikleri çekilir ve dataframe'den cıkarılır.
            
            min_max_scaler = preprocessing.MinMaxScaler()
            normalized_df = min_max_scaler.fit_transform(df.drop(['links','numberOfLikes'],axis= 1))
            
            x_test = df.drop('links',axis = 1).iloc[-1,:-1].values
            y_test = df.iloc[-1,-1]
            df.drop(df.tail(1).index, axis = 0,inplace = True)
            
            # Eger test videosu linklerin icinde hala varsa cıkarılır. Kanalın tek videosu varsa ? Dikkat et buna 
            rootIndex = -1
            rootURL = self.linkText.get().split("&")[0]
            for i,link in enumerate(df.loc[:,'links']):
                if link == rootURL:
                    rootIndex = i       
            if rootIndex != -1:
                df.drop(rootIndex,axis = 0,inplace = True)
                self.df.drop(rootIndex,axis = 0,inplace = True)
            
            self.progress['value']+=5
            df.drop('links', axis = 1,inplace = True) 
            x_train = np.array(df.drop('numberOfLikes',axis = 1))
            y_train = np.array(df.loc[:,'numberOfLikes'])
            
            predictions = []
            errors = []
            
            regressionLinear =  LinearRegression()
            regressionRidge =  Ridge()
            regressionRF =  RandomForestRegressor()
            regressionLasso =  Lasso()
            regressionMLP =  MLPRegressor()
            
            models = [regressionLinear,regressionRidge,regressionRF,regressionLasso,regressionMLP]
    
            for model in models :
                model.fit(x_train,y_train)
                pred = int(abs(model.predict(np.array(x_test).reshape(1,-1))))
                error = abs(y_test - pred) / (y_test+0.00000000001)
                errors.append(round(error,3))
                predictions.append(pred)
            
            self.progress['value']+=5
            self.results = {'Models':['Linear','Ridge','Random Forest','Lasso','Multi Layer Perceptron'],
                       'Predictions': predictions,
                       'Errors' : errors}
            
            self.results = pd.DataFrame(self.results).sort_values(by="Errors")
            self.showTablesOnScreen()
        else:
            self.progress.grid_forget()
            self.tables.pack_forget()
            for widget in self.tables.winfo_children():
                widget.destroy()
            self.progress['value'] = 0
            self.submit.grid(row = 1,column = 2)
            
            
    def showTablesOnScreen(self):
        style = ttk.Style()
        style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Helvetica', 11), rowheight=20)
        style.configure("mystyle.Treeview.Heading", font=('Helvetica', 13,'bold'))
        
        self.tableDF = ttk.Treeview(self.tables, style = "mystyle.Treeview",height = 10, selectmode='extended')
        self.tableRes = ttk.Treeview(self.tables, style = "mystyle.Treeview",height = 5, selectmode='extended')

        
        self.tableDF['show'] = 'headings'
        self.tableRes['show'] = 'headings'
        
        self.tableDF["columns"] = self.df.columns.values
        self.tableRes["columns"] = self.results.columns.values
        for x,col in enumerate(self.df.columns.values):
            self.tableDF.column(x, minwidth=0,width=150, stretch=NO, anchor = CENTER)
            self.tableDF.heading(x, text=col, anchor = CENTER)
            
        for x,col in enumerate(self.results.columns.values):
            self.tableRes.column(x, minwidth=0,width=300, stretch=NO, anchor = CENTER)
            self.tableRes.heading(x, text=col, anchor = CENTER)
        
        for i in range(len(self.df)):
            if i%2:
                self.tableDF.insert("", "end", text = "L"+str(i),values= self.df.iloc[i,:].tolist(),tags = ('odd',)) 
            else:
                self.tableDF.insert("", "end", text = "L"+str(i),values= self.df.iloc[i,:].tolist(),tags = ('even',)) 
            
        for i in range(len(self.results)):
            if i%2:
                self.tableRes.insert("", "end", values= self.results.iloc[i,:].tolist(),tags = ('odd',))
            else:
                self.tableRes.insert("", "end", values= self.results.iloc[i,:].tolist(),tags = ('even',))
         
        dfScrollBar = ttk.Scrollbar(self.tables, orient="vertical", command=self.tableDF.yview)
        dfScrollBar.grid(row = 0,column = 1,sticky = 'ns') 
        self.tableDF.configure(yscroll = dfScrollBar.set) 
        
        resultsScrollBar = ttk.Scrollbar(self.tables, orient="vertical", command=self.tableRes.yview)
        resultsScrollBar.grid(row = 1,column = 1,sticky = 'ns') 
        self.tableRes.configure(yscroll = resultsScrollBar.set) 
                
        self.tableDF.grid(row = 0,column = 0,sticky = 'ew')
        self.tableRes.grid(row =  1,column = 0,sticky = 'ew')
        
        self.tables.pack(fill = 'x')
        self.progress.grid_forget()
        self.submit.grid(row = 1,column = 2)
        
    def getDataFromURL(self):
        self.options = webdriver.chrome.options.Options()
        
        self.options.add_argument("--headless")
        self.options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.options.add_experimental_option('useAutomationExtension', False)
        self.options.add_argument('--window-size=1920,1080')
        self.options.add_argument('--disable-gpu')
        
        self.driver = webdriver.Chrome(options = self.options)
        self.progress['value']+=5
        time.sleep(1)
        
        url = self.linkText.get()
        try:
            #Videoya girilir.
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.XPATH,'//*[@id="text"]/a')))
        except:
            messagebox.showerror("Error", "This URL is broke. Please change it and try again.")
            return 
        
        self.progress['value']+=2
        
        #videoyu yayınlayan kanala gidilir. Daha sonra kanalın yayınladığı videolar sayfasına geçiş yapılır.
        channelLink = self.driver.find_element_by_xpath('//*[@id="text"]/a').get_attribute("href")
        channelUrl = channelLink+"/videos"
        self.driver.get(channelUrl)
        
        WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.XPATH,'//*[@href][@id="video-title"][@class="yt-simple-endpoint style-scope ytd-grid-video-renderer"]')))
        
        self.progress['value']+=2
        
        #Kanalın anasayfasında gösterilen videolara ait URL'ler ve yüklenme tarihleri toplanır.(İlk 30)
        urls = self.driver.find_elements_by_xpath('//*[@href][@id="video-title"][@class="yt-simple-endpoint style-scope ytd-grid-video-renderer"]')
        
        links = []
        for i in range(len(urls)):
            links.append(urls[i].get_attribute("href"))
            
        self.progress['value']+=1
        
        # Yeterli sayıda training sample yoksa islemler sonlanır.
        if len(links) < 5:
            messagebox.showerror("Error", "This URL does not seems to appropriate for this application. There are not enough train data. Please change URL and try again.") 
            return
        
        links.append(url) # dizinin son linki test videosunun linkidir.
        #  Bu linkler tek tek açılarak views,like,dislike ve comment sayıları toplanır.
        numberOfViews = []
        numberOfLikes = []
        numberOfDisLikes = []
        numberOfComments = []
        dates = []
        titles = []
        
        for i,url in enumerate(links):
            self.progress['value']+=70/len(links)
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.XPATH,'//*[@id="count"]/yt-view-count-renderer/span[1]')))
            self.driver.execute_script("window.scrollBy(0,500)")
            WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.XPATH,'//*[@id="count"]/yt-formatted-string')))
            
            videoTitle = self.driver.find_element_by_xpath('//*[@id="container"]/h1/yt-formatted-string').text
            titles.append(videoTitle)
            #Video görüntülenme sayisi
            numberOfViews.append(self.driver.find_element_by_xpath('//*[@id="count"]/yt-view-count-renderer/span[1]').text)
            
            likeSide = self.driver.find_elements_by_xpath('//*[@id="text"][@class="style-scope ytd-toggle-button-renderer style-text"]')
            #Video yorum sayisi
            numberOfComments.append(self.driver.find_element_by_xpath('//*[@id="count"]/yt-formatted-string').text)
            #Video yayınlanma tarihi
            dates.append(self.driver.find_element_by_xpath('//*[@id="metadata-line"]/span[2]').text)
            #Video beğeni sayisi
            numberOfLikes.append(likeSide[0].get_attribute("aria-label"))
            #Video beğenilmeme sayisi
            numberOfDisLikes.append(likeSide[1].get_attribute("aria-label"))
        
        self.driver.quit()
        dates_2 = dates.copy()
        
        numberOfViews = self.view_dislike_dislike_like_translation(numberOfViews)
        numberOfLikes = self.view_dislike_dislike_like_translation(numberOfLikes)
        numberOfDisLikes = self.view_dislike_dislike_like_translation(numberOfDisLikes)
        numberOfComments = self.view_dislike_dislike_like_translation(numberOfComments)
        dates = self.date_translation(dates)
        
        self.progress['value']+=5
        return numberOfViews,numberOfLikes,numberOfDisLikes,numberOfComments,dates_2,dates,titles,links
    #Youtube üzerinden çekilen verileri sayısal değerlere dönüştürür
    def view_dislike_dislike_like_translation(self,arr):
        for i,phrase in enumerate(arr):
            try:
                arr[i] = int(phrase.split(' ')[0].replace('.',''))
            except:
                arr[i] = 0
        return arr
    
    def date_translation(self, arr):
        for i,phrase in enumerate(arr):
            power = int(phrase.split(' ')[0])
            parameter = phrase.split(' ')[1]
            arr[i] =  power*self.parameters[parameter]
        return arr



gui = tkinter.Tk()
gui.geometry("1240x1080")
gui.title('YouTube Like Prediction')
gui.configure(background = '#282828')

my_gui = GUI(gui)
my_gui.screen()
gui.mainloop()
