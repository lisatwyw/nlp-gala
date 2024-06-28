

<details>
  <summary>Afraz's repo/notes - shared 2024-06-27 for potential use in PHP-L projects </summary>

 
  ## Dependencies
  
  The software is totally compiled in Python, it is possible to run it in both Python 2.x or Python 3.x but the code was written in Python 3.x. There are many libraries used for the webscraping portion of this exercise. They are:
   
  - Selenium
  
  - Beautiful Soup
  
  - Requests
  
  - Regex
  
  - Xlwt (Excel Support)
  
  - aiohttp (Asynchronous Requests)
  
  - country_list (for author country)
   
  Python has Requests and Regex installed by default but both selenium and Beautiful Soup need to be installed. Before installing any software kindly ensure that you have [pip](https://www.makeuseof.com/tag/install-pip-for-python/) installed.
   
  **READ CAREFULLY**: To save time run the file `setup.sh` which will install the dependencies for you
   
  ### Windows
   
  If you have Cmder then do the following steps:
   
  1) `$ chmod u+x setup.sh`
  
  2) `$ sh setup.sh`
   
  ### Mac/Linux
   
  1) `$ chmod u+x setup.sh`
  
  2) `$ ./setup.sh`
   
  ## Beautiful Soup
  
  To install Beautiful Soup on your system open your terminal (cmd) window and type:
   
  `$ pip install beautifulsoup4`
   
  It should install all the sub-dependencies like `xml` etc. on its own
   
  ## Selenium
  
  **UPDATE:** Firefox compatibility has also been added. See the relevant code blocks and repeat the steps here but use [geckodriver](https://github.com/mozilla/geckodriver/releases) for step 2. 
  
  This installation is quite difficult. As a result I have prepared a step by step guide:
   
  1. Ensure you have the **latest version** of Chrome. If not in Chrome click on `Settings > About Chrome`, it will automatically update itself. Ideally it should have Chrome 81 as of 9th April 2020. 
  
  2. Ensure you have the **latest version** of the Selenium Web Driver for [Chrome](https://sites.google.com/a/chromium.org/chromedriver/downloads). This repository already includes the latest version for Windows but depending on your system **you will need to install it in the same path as this repository**. Also note that if you do not have the latest version of Chrome then find out the version supported by your system under Step 1 and install the corresponding selenium driver. 
  
  3. After everything is in place in your terminal (cmd) window run `$ pip install selenium` and it should compile without any problems.
   
  ## Xlwt
   
  `$ pip install xlwt`
   
  ## aiohttp
   
  `$ pip install aiohttp`
   
  ## country_list
   
  `$ pip install country_list`
   
  ## PhantomJS requirement (Windows)
   
  The code is now set to work with 3 different browsers. PhantomJS is a headless browser and is a preferred method of using the script because it runs very fast. To ensure the phantomJS works change your phantomJS path in the script which can be found in the script on:
   
  `browser = webdriver.PhantomJS("C://Users//kingm//Desktop//UBC//ssrnextraction//phantomjs-2.1.1-windows//bin//phantomjs.exe")`
   
  You can locate your Path by typing `dir` on Windows or `ls` on Linux/Unix and you must modify this in the parameters in the string, there is no way to make this dynamic. The file to modify it on is `ssrnsearchoption.py`

</details>
