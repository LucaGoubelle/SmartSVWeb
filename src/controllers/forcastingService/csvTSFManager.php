<?php

    //author: Luca Goubelle
    define("PATH_ROOT", dirname(__FILE__));

    require_once PATH_ROOT ."/../../models/forcastingService/timeSerieForcastedChart.php";
    require_once PATH_ROOT ."/../../models/forcastingService/timeSerie.php";
    require_once PATH_ROOT ."/../../models/forcastingService/legend.php";
    require_once PATH_ROOT ."/../../models/forcastingService/forcasting.php";
    //require_once "csvTSManager.php";

    class CSV_TSF_Manager
    {
        //classic function to get the content of a csv file and put it in a array
        public static function arrayFromCSV($file, $hasFieldNames = false, $delimiter = ';', $enclosure='\\') 
        {
            ini_set("auto_detect_line_endings", true);
            $result = array();
            $size = filesize($file) +1;
            $file = fopen($file, 'r');
            if ($hasFieldNames) $keys = fgetcsv($file, $size, $delimiter, $enclosure);
            while ($row = fgetcsv($file, $size, $delimiter, $enclosure)) {
                $n = count($row); $res=array();
                for($i = 0; $i < $n; $i++) {
                    $idx = ($hasFieldNames) ? $keys[$i] : $i;
                    $res[$idx] = $row[$i];
                }
                $result[] = $res;
            }
            fclose($file);
            array_shift($result); 
            return $result;
        }

        public static function getTimeSerieForcastedFromArray($csvArr)
        {
            //to do : extract the timeSeries in array
            $a = array();
            $tsf = NULL;
            if(count($csvArr[0])==1)
            {
                foreach($csvArr as $arr)
                {
                    if(is_numeric($arr[0]))
                    {
                        $a[] = doubleval($arr[0]);
                    }
                }
                $tsf = new TimeSerie($a);
            }
            else if (count($csvArr[0])==2)
            {
                foreach($csvArr as $arr)
                {
                    if(is_numeric($arr[1]))
                    {
                        $a[] = doubleval($arr[1]);
                    }
                }
                $tsf = new TimeSerie($a);
            }
            return $tsf;
        }

        public static function getDatesOrLegendFromField($csvArr)
        {
            $legendArr = array();
            $doNotMatter = FALSE;
            foreach($csvArr as $arr)
            {
                if(DateTime::createFromFormat('d/m/Y H', $arr[0])!==FALSE)
                {
                    $legendArr[] = $arr[0];
                }
                else if(doubleval($arr[0])==FALSE)
                {
                    $legendArr[] = $arr[0];
                }
                else
                {
                    $doNotMatter = TRUE;
                }
            }
            if($doNotMatter==TRUE)
            {
                $legend = new Legend($legendArr, $matter="no");
            }
            else
            {
                $legend = new Legend($legendArr);
            }
            return $legend;
        }


        //*** CALLABLE METHODS ***
        // 
        public static function getTSF_inCSV($filePath)
        {
            $csvArr = CSV_TSF_Manager::arrayFromCSV($filePath);
            return CSV_TSF_Manager::getTimeSerieForcastedFromArray($csvArr);
        }

        public static function getLegend_inCSV($filePath)
        {
            $csvArr = CSV_TSF_Manager::arrayFromCSV($filePath);
            return CSV_TSF_Manager::getDatesOrLegendFromField($csvArr);
        }

        //get a time serie graph from a csv file
        /*public static function getTSF_Graph_fromCSV($filePath)
        {
            $csvArr = CSV_TSF_Manager::arrayFromCSV($filePath);
            return new TimeSerieForcastedChart(CSV_TSF_Manager::getDatesOrLegendFromField($csvArr), CSV_TSF_Manager::getTimeSerieForcastedFromArray($csvArr));
        }*/

        
    }

    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------
    #---------------------------------------------------------------------------------------------

    class CSV_MTSF_Manager
    {
        //classic function to get the content of a csv file and put it in a array
        public static function arrayFromCSV($file, $hasFieldNames = false, $delimiter = ';', $enclosure='\\') 
        {
            ini_set("auto_detect_line_endings", true);
            $result = array();
            $size = filesize($file) +1;
            $file = fopen($file, 'r');
            if ($hasFieldNames) $keys = fgetcsv($file, $size, $delimiter, $enclosure);
            while ($row = fgetcsv($file, $size, $delimiter, $enclosure)) {
                $n = count($row); $res=array();
                for($i = 0; $i < $n; $i++) {
                    $idx = ($hasFieldNames) ? $keys[$i] : $i;
                    $res[$idx] = $row[$i];
                }
                $result[] = $res;
            }
            fclose($file);
            array_shift($result); 
            return $result;
        }

        public static function isLegend($csvArr)
        {
            if(is_numeric($csvArr[1][0]))
            {
                return TRUE;
            }
            else
            {
                return FALSE;
            }
        }

        public static function getNbTS($csvArr)
        {
            if(count($csvArr[0])==1)
            {
                return 1;
            }
            else if(count($csvArr[0])==2 AND isLegend($csvArr))
            {
                return 1;                
            }
            else if(count($csvArr[0])==2 AND !isLegend($csvArr))
            {
                return 2;
            }
            else if(count($csvArr[0])>2 AND isLegend($csvArr))
            {
                return count($csvArr[0])-1;
            }
            else if(count($csvArr[0])>2 AND !isLegend($csvArr))
            {
                return count($csvArr[0]);
            }
        }

        public static function getForcastedSerieFromArray($csvArr)
        {
            $aFS = array();
            $a = array();
            if(count($csvArr[0])==1)
            {
                foreach($csvArr as $arr)
                {
                    if(is_numeric($arr[0]))
                    {
                        $a[] = doubleval($arr[0]);
                    }
                }
                $aFS[] = new TimeSerie($a);
            }
            else
            {
                for($i=0;$i<count($csvArr[0]);$i++){ $a[] = array(); }
                
                for($i=0;$i<count($csvArr);$i++)
                {
                    for($j=0;$j<count($csvArr[$i]);$j++)
                    {
                        if(is_numeric($csvArr[$i][$j]))
                        {
                            $a[$j][] = doubleval($csvArr[$i][$j]);
                        }
                    }
                }
                for($k=0;$k<count($a);$k++)
                {
                    $aFS[] = new TimeSerie($a[$k]);
                }
            }
            return $aFS;
        }

        public static function getDatesOrLegendFromField($csvArr)
        {
            $legendArr = array();
            $doNotMatter = FALSE;
            foreach($csvArr as $arr)
            {
                if(DateTime::createFromFormat('d/m/Y H', $arr[0])!==FALSE)
                {
                    $legendArr[] = $arr[0];
                }
                else if(doubleval($arr[0])==FALSE)
                {
                    $legendArr[] = $arr[0];
                }
                else
                {
                    $doNotMatter = TRUE;
                }
            }
            if($doNotMatter==TRUE)
            {
                $legend = new Legend($legendArr, $matter="no");
            }
            else
            {
                $legend = new Legend($legendArr);
            }
            return $legend;
        }


        //*** CALLABLE METHODS ***
        //
        public static function getMTSF_inCSV($filePath)
        {
            $csvArr = CSV_MTSF_Manager::arrayFromCSV($filePath);
            return CSV_MTSF_Manager::getForcastedSerieFromArray($csvArr);
        }

        public static function getLegend_inCSV($filePath)
        {
            $csvArr = CSV_MTSF_Manager::arrayFromCSV($filePath);
            return CSV_MTSF_Manager::getDatesOrLegendFromField($csvArr);
        }
    }
?>