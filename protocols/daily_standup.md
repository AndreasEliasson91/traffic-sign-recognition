**Mall "daily standup"**

## Daily standup 2022-05x17

#### Abshir
* kollat på sätt att göra visualization funktion 
* få ner det i kod
* Behöver du hjälp med något?  nej

#### Andreas
* har gjort en intial prediction
* fortsätta med 
* Behöver du hjälp med något?  nej

#### Fredrik
frånvarande / jobb  

#### Kajsa
* jobbat med en mip 25 
* fortsätta 
* Behöver du hjälp med något?  

## Daily standup 2022-xx-xx

#### Abshir
* Vad har du gjort sedan förra avstämningen?  
* Vad ska du göra till nästa avstämning?  
* Behöver du hjälp med något?  

#### Andreas
* Vad har du gjort sedan förra avstämningen?  
* Vad ska du göra till nästa avstämning?  
* Behöver du hjälp med något?  

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
* Vad ska du göra till nästa avstämning?  
* Behöver du hjälp med något?  

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
* Vad ska du göra till nästa avstämning?  
* Behöver du hjälp med något?  

_____________________________________________________________________________________________
## Daily standup 2022-05-16

#### Abshir
* Vad har du gjort sedan förra avstämningen?  
Har tittat på MIP-67, problem med att koden kraschar när det är två bilder på samma skylt. Issuen är ännu inte löst, och vi bestämmer att avvakta med den för tillfället, eftersom det fungerar när vi kör koden på Colab.  
* Vad ska du göra till nästa avstämning?  
Ska jobba med MIP-74 Create visualization function (tillsammans med Fredrik).
* Behöver du hjälp med något?  
Nej.

#### Andreas
* Vad har du gjort sedan förra avstämningen?  
Stängt issue om optimering (MIP-62 med subtasks) efter att ha konstaterat att vi verkar vara nere på rimlig tid, ca 10 min per epoch för 2000 bilder.  
Kollat MIP-68 om problem med tensorvalue. Felet skulle lösas om vi byter från OpenCV till PIL. Har inte ändrat, pga att man måste ändra på så många platser.
På Colab körs en egen variant av cv2 (cv2_imshow) vilket gör att det funkar där. Vi bestämmer att släppa issue MIP-68 nu, så får vi se om vi måste uppdatera koden för att inte få problem i senare steg.
* Vad ska du göra till nästa avstämning?  
Admin, fixa till sprint review-anteckningarna. Sedan MIP-70 Build predict function.
* Behöver du hjälp med något?  
Nej.

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
Tränat modellen 15 epoker. Vi börjar få en stabil loss, men kurvan är lite taggig. Vi tar med oss detta till Joakim på onsdag. 
Städat och uppdaterat datan samt annoteringarna. Har lagt till spegelvända versioner av bilderna för att få fler bilder. Undersökt lite om optimering av data. 
* Vad ska du göra till nästa avstämning?  
Dela pth-filen (MIP-76). Kommer inte hinna göra jättemycket mer, men börjar titta på MIP-74 Create visualization function.
* Behöver du hjälp med något?  
Nej.

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
Åkt tåg... Läst kursbok samt om PIL vs OpenCV kopplat till MIP-68.
* Vad ska du göra till nästa avstämning?  
Städa upp daily-anteckningar. Sedan börja titta på create input from file (MIP-25).
* Behöver du hjälp med något?  
Nej.

#### Övrigt
Vi bestämmer att köra daily kl 11.30 den här veckan, så kan Abshir också vara med. Fredrik kommer antagligen inte kunna vara med på daily pga jobb minst tis-tors.
_______________________________________________________________________________


## Daily standup 2022-05-12

#### Abshir
Frånvarande 

#### Andreas
* Vad har du gjort sedan förra avstämningen?  
Städat och strukturerat om i projektet. MIP-69.
Tillsammans med Kajsa jobbat med att få in RCNN-notebooken i projektet (MIP-65).
* Vad ska du göra till nästa avstämning?  
Se om vi kan optimera bilderna för att få upp hastigheten (MIP-58)
* Behöver du hjälp med något?  
Nej. Eventuellt sedan, om det är något problem med bilderna.

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
Mergat in senaste version av LU-datan. Har processat bilderna lite, bl a för att få ner storleken. Kör just nu träning på det nya datasettet.
* Vad ska du göra till nästa avstämning?  
Kör just nu träning på den nya versionen av datasetet, samt utforskar optimeringsmöjligheter (MIP-62)
* Behöver du hjälp med något?  
Nej, inte just nu!

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
Jobbat med att få in RCNN-notebooken i projektet (MIP-65).
* Vad ska du göra till nästa avstämning?  
Buggfixar, MIP-67 och MIP-68.
* Behöver du hjälp med något?  
Nej.

#### Övrigt
Vi bestämmer att merga in den strukturen som Andreas har fixat med (MIP-69, branch restructured).
______________________________________________________________________________


## Daily standup 2022-05-11

#### Andreas
* kollade på rcnn tutorial.
* kolla om det går att optimera hastigheten på training


#### Abshir
* kollade på att träna data snabbare.
* undersöka varför det blir svarta bilder 


#### Fredrik
*jobbar


#### Kajsa
* jobbade med mip 65
* fortsätta med den under dagen
__________________________________________________________


## Daily standup 2022-05-05

#### Andreas
* Vad har du gjort sedan förra avstämningen?
Tweakade siffror i CNN-modellen för att få ett eventuellt bättre resultat. Började kolla hur man kan använda CNN-modellen för object detection och fann en tutorial för RCNN som vi kollar vidare på.

* Vad ska du göra till nästa avstämning?
Kolla upp Object Detection med RCNN

* Behöver du hjälp med något?  
nej

#### Abshir
* Vad har du gjort sedan förra avstämningen?
Läste på om Yolov5

* Vad ska du göra till nästa avstämning?
Fortsätta testa Yolov5 och bygga ett litet test.
(Kolla upp Object Detection med RCNN)
 
* Behöver du hjälp med något?
Nej

#### Fredrik
* Vad har du gjort sedan förra avstämningen?
Färdigställt Linköping-setet 1.0 och teskör det i Detecto under dagen. Konverterat annotation-filen till xml (fungerar bra med RCNN-algoritmen vi ska testa)

* Vad ska du göra till nästa avstämning?
Testkör LU-setet i Detecto under dagen
Kolla upp Object Detection med RCNN

* Behöver du hjälp med något?
Nej

#### Kajsa
* Vad har du gjort sedan förra avstämningen?
Labbade med "Hidden Layer". Kom fram till att det tyvärr är lite outvecklat och mycket småfix/problemlösning av smågrejer

* Vad ska du göra till nästa avstämning?
Fortsätta från morgondagen och gärna få det att lyckas. Funkar det, så funkar det (inget akut behov, men kul om det lyckas)
Kolla upp Object Detection med RCNN

* Behöver du hjälp med något?
Nej
________________________________________________________________________


## Daily standup 2022-05-04

#### Andreas

* Vad har du gjort sedan förra avstämningen?
Testade att få numeriska värden till CNN-modellen och se om man får samma resultat

* Vad ska du göra till nästa avstämning?
Kolla vidare på CNN och se om man kan tweaka vår modell

* Behöver du hjälp med något?  
nej

#### Abshir
* Vad har du gjort sedan förra avstämningen?
Läste på om CNN

* Vad ska du göra till nästa avstämning?
 Tar en genomgång med Fredrik om Linköping-setets uppbyggnad. Kolla om man ska konvertera csv-filen (annotation) till xml för att underlätta detecto
 Utforska Yolo5
 
* Behöver du hjälp med något?
Nej

#### Fredrik
* Vad har du gjort sedan förra avstämningen?
Kollat detecto 

* Vad ska du göra till nästa avstämning?
Kolla vidare på detecto och Linköping-setet. Kolla om man ska konvertera csv-filen (annotation) till xml

* Behöver du hjälp med något?
Nej

#### Kajsa
* Vad har du gjort sedan förra avstämningen?
Kört lite inläsning igår

* Vad ska du göra till nästa avstämning?
Utforska hur man kan visualisera de dolda lagrena i ett Neuralt Nätverk, lägger till en JIRA-Issue

* Behöver du hjälp med något?
Nej


## Daily standup 2022-05-11

#### Andreas
* fixade med strukturen och lite bugfix
* idag ska jag kolla på mip 68


#### Abshir 

* jbba på mip 67


#### Fredrik
* frånvarande jobbar


#### Kajsa
* frånvarande


___________________________________________________________________


## Daily standup 2022-05-02

#### Fredrik
* Vad har du gjort sedan förra avstämningen?
* Jobbat med CSV-filen, och jobbat med och att parsa igenom den.
* Vad ska du göra till nästa avstämning?  förtsätter med samma uppgift


#### Andreas
* Vad har du gjort sedan förra avstämningen? Jobbat vidare med sin CNN modell. 
* Vad ska du göra till nästa avstämning?  fortsätter med samma uppgift


#### Övrigt
Tar sprint-planning 2022-05-03
_______________________________________________________________________


## Daily standup 2022-04-29

#### Andreas

 * Vad har du gjort sedan förra avstämningen?
Kört en tutorial på dataset, med CNN, med att omvandla till numeriska värden.
* Vad ska du göra till nästa avstämning?
fortsätter på samma spår
* Behöver du hjälp med något?  
nej

#### Abshir
* Vad har du gjort sedan förra avstämningen?  kollat youtube, och letat efter tutorials på maskning
* Vad ska du göra till nästa avstämning?  ska börja bygga en notebook med maskninsfunktioner. 
* Behöver du hjälp med något?  inget än

#### Fredrik
* Vad har du gjort sedan förra avstämningen?
kollat upp hur Colabs funkar, och rensat bort ännu mer nolldata, och felaktiga skyltar i dataset.

* Vad ska du göra till nästa avstämning?
Forksat vidare på Detecto och PyVision
* 
* Behöver du hjälp med något?
Nej

#### Kajsa
Ej tillgänglig
______________________________________________________________________


## Daily standup 2022-04-27

#### Andreas
* Vad har du gjort sedan förra avstämningen?
Testade att köra 'Detecto' på wiki-datasetet. 
Detta fungerar inte då det är hela bilden som är skylten, så gick över till Pytorchs version 
och kom till samma insikt där, vi kommer behöva maska bilderna innan vi kan testa igen.

* Vad ska du göra till nästa avstämning?  
Fortsätter kolla på image classification, och börjar med kategorisering.
* Behöver du hjälp med något?  
Nej, men sätter sig med Kajsa för att bolla idéer.

#### Alexander
Frånvarande


#### Abshir
* Vad har du gjort sedan förra avstämningen?
Klar med sin issue att maskera bilder, ska mergas in.
* Vad ska du göra till nästa avstämning?  
Ska se över om det gör att använda Pyvision med maskade bilder. 
* Behöver du hjälp med något?  
Vet inte ännu

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
Filtrerat alla bilderna, och annoterat upp Varnings-skyltar
* Vad ska du göra till nästa avstämning?  
Skriva om CSV-fil, och eventuellt formatera om den, så att den passar Detecto
* Behöver du hjälp med något?  
Nej

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
MIP-19. Jobbat med att identifiera konturer, har även testat "template matching" detta funkar med identiska pixlar och färger men KAN vara något att sätta sig in i. Släppt det just nu. 
* Vad ska du göra till nästa avstämning?  
Ska börja med MIP-40, och se om det fungerar på wiki-datan. Ska även sätta sig med Andreas och bolla idéer.
* Behöver du hjälp med något?
Nej

#### Övrigt
* Andreas missar morgondagens möte, då han har bokad veterinärtid.
* Kajsa ska hämta ny jobbdator, och kommer inte närvara Fredag - måndag
* Alexander faller bort Tors - måndag pga. Danmarksresa
_________________________________________________________________


## Daily standup 2022-04-22

#### Andreas
* Vad har du gjort sedan förra avstämningen?  
Löste problemet med färger på skyltarna.
* Vad ska du göra till nästa avstämning?
Fixa till mappstruktur för bilderna. Ska även dela upp wikipediadatat i tre olika set. Börjar sedan med MIP-16, att förbättra annotations för wikipediadatan.  
* Behöver du hjälp med något?  
Nej.

#### Alexander
* Vad har du gjort sedan förra avstämningen?  
Var mest upptagen med jobb igår. Har fortsatt med att hitta underlag till beslut om presentationsgränssnitt (MIP-23). 
* Vad ska du göra till nästa avstämning?  
Göra klart runt beslutsunderlag om presentationsgränssnitt (MIP-23). Lägger upp info under dagen som alla kan titta på inför måndag.
* Behöver du hjälp med något?  
Nej.

#### Abshir
* Vad har du gjort sedan förra avstämningen?  
Har kollat på bildinput via kameran (MIP-14), missförstånd att den inte var med i veckans sprint.
* Vad ska du göra till nästa avstämning? 
Titta på möjlighet att känna igen skyltar baserat på att maska färger (MIP-20)
* Behöver du hjälp med något?  
Nej.

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
Sitter med Linköpingsdatan och håller på att plocka fram ett träningsset med ett antal tydliga skyltar. Jobbar också med formatet på annotations. Har guidat Britta till Linköpingsdatat.
* Vad ska du göra till nästa avstämning?  
Läsa in underlag om presentationsgränssnitt. Fortsätta med Linköpingsdatat (MIP-21, MIP-22).
* Behöver du hjälp med något?  
Beslut om struktur på annotationsdatan: Gruppen diskuterar och kommer fram till att man nog kan strukturera det så att bilder är en klass och att signs är en lista i klassen som innehåller ett dictionary för varje skylt.

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
Jobbat med bildigenkänning med hjälp av former (MIP-19).
* Vad ska du göra till nästa avstämning?  
Ska fortsätta med MIP-19
* Behöver du hjälp med något?  
Nej.

#### Britta
Frånvarande

#### Övrigt
* Mötet på måndag blir kl 11 som preliminär tid. Alexander meddelar om tiden behöver justeras ytterligare. Kajsa är helt frånvarande på måndag.
* Inför måndagsmötet läser alla in sig på underlag om presentationsgränssnitt.
* Fredrik tar över stafettpinnen med att skriva mötesanteckningar inför nästa vecka.
__________________________________________________________


## Daily standup 2022-04-21

#### Andreas
* Vad har du gjort sedan förra avstämningen?  
Fixat projektstrukturen (MIP-15).
* Vad ska du göra till nästa avstämning?  
Lösa problem från fix av projektstrukturen. Jobba på att få ordning på färgerna i wikipedia-datan (MIP-28).
* Behöver du hjälp med något?  
Nej.

#### Alexander
* Vad har du gjort sedan förra avstämningen?  
Undersökt varianter på GUI/presentationslager (MIP-23).
* Vad ska du göra till nästa avstämning?  
Sammanställa info om GUI/presentationslager (MIP-23).
* Behöver du hjälp med något?  
Tänkte prata med Fredrik om GUI-erfarenheter.

#### Abshir
* Vad har du gjort sedan förra avstämningen?
x
* Vad ska du göra till nästa avstämning?  
Ska kolla igenom sprintbacklogen och se vad som kan vara lämpligt.
* Behöver du hjälp med något?  
Nej.

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
Har varit i kontakt med Trafikverket (MIP-10). 
* Vad ska du göra till nästa avstämning?  
Jobbar vidare med att göra subsets av Linköpingsdatasetet (MIP-21) och tar även MIP-22 att strukturera datasetet.
* Behöver du hjälp med något?  
Nej.

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
Inläsning, labbat med att hitta konturer.
* Vad ska du göra till nästa avstämning?  
Tar subtask MIP-19, att hitta konturer i data.  
* Behöver du hjälp med något?  
Nej.

#### Britta
* Vad har du gjort sedan förra avstämningen?  
Kom just in i projektet och har jobbat med att läsa in sig på vad som har hänt.
* Vad ska du göra till nästa avstämning?  
Fortsätta läsa in sig på projektet och resurser.
* Behöver du hjälp med något?  
Fredrik introducerar Britta till Linköpingsdatasetet.

#### Övrigt
* Vi bestämmer att måndag blir deadline för att ta beslut om vilken väg vi ska gå gällande användargränssnitt. Alexander lägger upp info om detta på Discord och alla får i uppgift att läsa in sig/fundera tills på måndag.
* Vi bestämmer att släppa punkten att få hjälp av Trafikverket med data. Linköpingsdatasetet ser ut att fungera bra.
* Vi bestämmer att sprintar löper måndag-måndag. Det innebär att veckans sprint i praktiken blir väldigt kort.
_____________________________________________________________


## Daily standup 2022-04-20

#### Andreas
* Vad har du gjort sedan förra avstämningen?   
  Skapat träningsdata utifrån wikipediabilder.
* Vad ska du göra till nästa avstämning?  
  Nästan ingen arbetstid till nästa avstämning, då föreläsning + sprintplanering tog nästan hela dagen.
* Behöver du hjälp med något?  
  Det är något skumt med färgåtergivningen på vissa bilder, se dialog på Discord.

#### Alexander
Frånvarande

#### Abshir
Frånvarande

#### Fredrik
Frånvarande

#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
  Admin med mötesnoteringar, grävt lite i Linköpings-datan, tittat på problemet med färgåtergivning i bilder.  
* Vad ska du göra till nästa avstämning?  
  Nästan ingen arbetstid till nästa avstämning, då föreläsning + sprintplanering tog nästan hela dagen.
* Behöver du hjälp med något?  
  Nej  
_________________________________________________________________________


## Daily standup 2022-04-19

#### Andreas
* Vad har du gjort sedan förra avstämningen?
* Vad ska du göra till nästa avstämning?  
  Fortsätta med inititalt dataset.
* Behöver du hjälp med något?  
  Nej

#### Alexander
* Vad har du gjort sedan förra avstämningen?  
  Inläsning
* Vad ska du göra till nästa avstämning?  
  Ta fram alternativ på gränssnitt
* Behöver du hjälp med något?  
  Nej. Fredrik säger att han har jobbat en del med gränssnitt i tidigare projekt, och gärna bidrar som bollplank vid behov.

#### Abshir
Frånvarande

#### Fredrik
* Vad har du gjort sedan förra avstämningen?  
  Jobbat med att ta fram ett första dataset med miljöbilder på skyltar.
* Vad ska du göra till nästa avstämning?  
  Fortsätta med dataset med miljöbilder.
* Behöver du hjälp med något?  
  Det är en utmaning att få bilderna i samma upplösning, vilket krävs för att kunna analysera dem. Ska kolla med Trafikverket om de har några bilder man kan använda.
  Ska också kolla med Joakim om det är ok att vi använder Linköpings universitets annoterade dataset. Det innehåller information om vilka bilder som innehåller skyltar, och var i bilden skylten syns.
  
#### Kajsa
* Vad har du gjort sedan förra avstämningen?  
  Inläsning + lite administration runt mötesanteckningar.
* Vad ska du göra till nästa avstämning?  
  Fortsatt inläsning + gräva i Linköpingsdatasetet.
* Behöver du hjälp med något?  
  Nej.
  
#### Övrigt
* Det är svårt att fylla på backloggen inför veckans sprint utan mer kunskap om hur vi ska gå tillväga. Vi bestämmer att invänta Joakims lektion imorgon innan vi slutför veckans sprintplanering.