Cel:
Celem projektu jest analiza rynku nieruchomości i predykcja ceny za metr kwadratowy mieszkań na podstawie ich cech, takich jak powierzchnia, liczba pokoi, województwo i obecność miejsca parkingowego.

Zakres analizy:
Przetwarzanie i eksploracja danych o mieszkaniach.
Budowa i porównanie modeli regresji liniowej oraz Random Forest.
Ocena skuteczności modeli na podstawie metryk błędu i walidacji krzyżowej.
Wizualizacja wyników oraz identyfikacja najważniejszych cech wpływających na cenę.

Źródło danych:
Dane pobrane z platformy Kaggle pochodzą z portalu Otodom i zawierają szczegółowe informacje o mieszkaniach.
Link do danych: https://www.kaggle.com/datasets/amrahhasanov23/otodom-pl-flat-prices-in-poland

Projekt wykorzystuje warstwę shapefile to wizualizacji danych przestrzennych pozyskaną ze strony GIS Supportu.
Link do warstwy shapefile: https://gis-support.pl/baza-wiedzy-2/dane-do-pobrania/granice-administracyjne/

Przetwarzanie danych:
Usunięcie brakujących wartości.
One-hot encoding dla zmiennych kategorycznych (Voivodeship, Parking_Space, Number_of_Rooms).
Normalizacja ceny za metr kwadratowy jako wartości docelowej (Price_per_m2)

Zastosowane modele:
Regresja Liniowa – prosty model zakładający liniową zależność między cechami a ceną.
Random Forest Regressor – model zespołowy oparty na wielu drzewach decyzyjnych, dobrze radzący sobie z nieliniowościami.

Podział danych:
80% danych do trenowania modelu.
20% danych do testowania modelu.

Walidacja krzyżowa:
5-krotna walidacja krzyżowa została zastosowana do oceny stabilności modelu.

Wnioski:
Random Forest osiągnął znacznie lepsze wyniki niż regresja liniowa.
Powierzchnia i liczba pokoi są kluczowe dla ceny mieszkania.
Cena mniejszych mieszkań za metr kwadratowy jest wyższa niż większych mieszkań.
Najbardziej powszechne są oferty dotyczące mieszkań 2 i 3 pokojowych.

Aby uruchomić wizualizację należy użyć linku:
https://kfzemcvdcnk5l8kfldpevo.streamlit.app/
