import streamlit as st
import os

st.write('Below are links to the minutes accessible on colwood.civicweb.net')
links = ['https://colwood.civicweb.net/filepro/document/226140/Council%20-%2027%20May%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/225759/Council%20-%2013%20May%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/225757/Special%20Council%20-%2008%20May%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/226187/Special%20Infrastructure%20Committee%20-%2007%20May%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/225106/Special%20Council%20-%2029%20Apr%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/225103/Council%20-%2022%20Apr%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/224103/Council%20-%2008%20Apr%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/224642/Planning%20and%20Land%20Use%20Committee%20-%2002%20Apr%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/223073/Council%20-%2025%20Mar%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/225853/Emergency%20Planning%20Committee%20-%2019%20Mar%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/223793/Council%20-%2011%20Mar%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/223119/Planning%20and%20Land%20Use%20Committee%20-%2004%20Mar%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/222607/Council%20-%2026%20Feb%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/222211/Active%20Transportation%20Committee%20-%2020%20Feb%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/224796/Special%20Infrastructure%20Committee%20-%2013%20Feb%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/221675/Council%20-%2012%20Feb%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/221383/Planning%20and%20Land%20Use%20Committee%20-%2005%20Feb%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/220962/Public%20Hearing%20-%2025%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/221035/Council%20-%2022%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/221033/Special%20Council%20-%20Budget%20Deliberations%20-%2016%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/223200/Emergency%20Planning%20Committee%20-%2016%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/220402/Special%20Council%20-%20Budget%20Deliberations%20-%2011%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/220405/Special%20Council%20-%20Budget%20Deliberations%20-%2009%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/220400/Council%20-%2008%20Jan%202024%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/219255/Special%20Council%20-%20Service%20Review%20-%2013%20Dec%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/221295/Special%20Environment%20Committee%20-%2013%20Dec%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/219247/Council%20-%2011%20Dec%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/219253/Special%20Council%20-%20Service%20Review%20-%2007%20Dec%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/220630/Planning%20and%20Land%20Use%20Committee%20-%2004%20Dec%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/219251/Special%20Council%20-%20Service%20Review%20-%2004%20Dec%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/219249/Special%20Council%20-%20Service%20Review%20-%2029%20Nov%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/218476/Council%20-%2027%20Nov%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/220412/Emergency%20Planning%20Committee%20-%2021%20Nov%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/218538/Special%20Environment%20Committee%20-%2020%20Nov%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/218148/Council%20-%2014%20Nov%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/218230/Planning%20and%20Land%20Use%20Committee%20-%2006%20Nov%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/224240/Waterfront%20Stewardship%20Committee%20-%2026%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/217418/Council%20-%2023%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/221251/Active%20Transportation%20Committee%20-%2016%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/217403/Environment%20Committee%20-%2016%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216641/Council%20-%2010%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/217168/Planning%20and%20Land%20Use%20Committee%20-%2003%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/223111/Parks%2C%20Trails%2C%20and%20Recreation%20Committee%20-%2003%20Oct%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/225476/Special%20Heritage%20Commission%20-%2028%20Sep%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216487/Council%20-%2025%20Sep%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/219177/Emergency%20Planning%20Committee%20-%2019%20Sep%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216086/Council%20-%2011%20Sep%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216662/Planning%20and%20Land%20Use%20Committee%20-%2005%20Sep%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216084/Council%20-%2028%20Aug%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/215084/Council%20-%2010%20Jul%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/215648/Planning%20and%20Land%20Use%20Committee%20-%2004%20Jul%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/214324/Council%20-%2026%20Jun%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216666/Active%20Transportation%20Committee%20-%2019%20Jun%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216668/Environment%20Committee%20-%2019%20Jun%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/213917/Council%20-%2012%20Jun%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216664/Special%20Parks%2C%20Trails%2C%20and%20Recreation%20Committee%20-%2012%20Jun%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/213515/Council%20-%2023%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216726/Special%20Waterfront%20Stewardship%20Committee%20-%2018%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216047/Emergency%20Planning%20Committee%20-%2016%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212905/Special%20Council%20-%2015%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212903/Special%20Council%20-%2011%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/216449/Heritage%20Commission%20-%2011%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212872/Council%20-%2008%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/214165/Planning%20and%20Land%20Use%20Committee%20-%2001%20May%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212025/Special%20Council%20-%2027%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212795/Waterfront%20Stewardship%20Committee%20-%2027%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/211696/Special%20Council%20-%2026%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/211441/Board%20of%20Variance%20-%2025%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212023/Council%20-%2024%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/213701/Active%20Transportation%20Committee%20-%2017%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/213703/Environment%20Committee%20-%2017%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/211299/Council%20-%2011%20Apr%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212063/Board%20of%20Variance%20-%2028%20Mar%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/210725/Council%20-%2027%20Mar%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/212329/Emergency%20Planning%20Committee%20-%2021%20Mar%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/210445/Council%20-%2013%20Mar%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/209958/Public%20Hearing%20-%2002%20Mar%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/210420/Board%20of%20Variance%20-%2028%20Feb%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/210137/Council%20-%2027%20Feb%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/209655/Special%20Council%20-%20Budget%20Deliberations%20-%2023%20Feb%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/210135/Special%20Council%20-%20Budget%20Deliberations%20-%2021%20Feb%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/209661/Council%20-%2013%20Feb%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208959/Special%20Council%20-%2030%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208957/Special%20Council%20-%20Service%20Review%20-%2024%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208955/Council%20-%2023%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208953/Special%20Council%20-%20Service%20Review%20-19%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208951/Special%20Council%20-%20Service%20Review%20-%2017%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/210379/Emergency%20Planning%20Committee%20-%2017%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208949/Special%20Council%20-%20Service%20Review%20-%2012%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/208947/Special%20Council%20-%20Service%20Review%20-%2010%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/206672/Council%20-%2009%20Jan%202023%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/205839/Council%20-%2012%20Dec%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/205837/Special%20Council%20-%2005%20Dec%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/204588/Council%20-%2028%20Nov%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/211285/Board%20of%20Variance%20-%2022%20Nov%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/206427/Emergency%20Planning%20Committee%20-%2015%20Nov%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/204361/Council%20-%2014%20Nov%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/203767/Inaugural%20-%2007%20Nov%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/203769/Council%20-%2011%20Oct%202022%20-%202_30%20pm%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/202552/Council%20-%2026%20Sep%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/209740/Public%20Hearing%20-%2022%20Sep%202022%20Minutes.pdf',
 'https://colwood.civicweb.net/filepro/document/204520/Emergency%20Planning%20Committee%20-%2020%20Sep%202022%20Minutes.pdf']


for l in links:  
 txt = os.path.basename( l )
 txt = txt.replace('%20', ' ')
 st.markdown( '[%s](%s)' % (txt, l) )