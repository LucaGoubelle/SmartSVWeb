<?php

    //author: Luca Goubelle
	define("PATH_ROOT", dirname(__FILE__));
	

    require_once PATH_ROOT ."/../../models/forcastingService/timeSerie.php";
    require_once PATH_ROOT ."/../../models/forcastingService/legend.php";

    class CSV_MTS_Manager
    {
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
            $ind = array_shift($result); 
            return array($ind, $result);
        }

        public static function getTimeSerieFromArray($csvArr)
        {
            array_shift($csvArr[0]);
            $indexis = $csvArr[0];
            $csvArrValues = $csvArr[1];
            $a = array();
            $aTS = array();
            if(count($csvArrValues[0])==2)
            {
                foreach($csvArrValues as $arr)
                {
                    if(is_numeric($arr[1]) OR $arr[1]=="")
                    {
                        $a[] = doubleval($arr[1]);
                    }
                    else if(!is_numeric($arr[1]))
                    {
                        $a[] = $arr[1];
                    }
                }
                $aTS[] = new TimeSerie($a);
            }
            else
            {
                for($i=0;$i<count($csvArrValues[0])-1;$i++){ $a[] = array(); }
                
                for($i=0;$i<count($csvArrValues);$i++)
                {
                    for($j=1;$j<count($csvArrValues[$i]);$j++)
                    {
                        if(is_numeric($csvArrValues[$i][$j]) OR $csvArrValues[$i][$j]=="")
                        {
                            $a[$j-1][] = doubleval($csvArrValues[$i][$j]);
                        }
                        else if(!is_numeric($csvArrValues[$i][$j]))
                        {
                            $a[$j-1][] = $csvArrValues[$i][$j];
                        }
                    }
                }
                for($k=0;$k<count($a);$k++)
                {
                    $aTS[] = new TimeSerie($a[$k]);
                }
            }

            for($i=0;$i<count($aTS);$i++)
            {
                if(!$aTS[$i]->isNumericColumn())
                {
                    array_splice($aTS, $i, 1);
                    array_splice($indexis, $i, 1);
                }
            }

            return new TimeSerieCollection($indexis,$aTS);
       
        }

        /* get the legend for the graph */
        public static function getDatesOrLegendFromField($csvArr)
        {
            $legendArr = array();
            foreach($csvArr as $arr)
            {
                    $legendArr[] = $arr[0];
            }
            return $legendArr;

        }

        //*** CALLABLE METHODS ***

        // Static member that return TimeSerie Object, the timeseries are took from a csv
        // file specified
        public static function getMTS_inCSV($filePath)
        {
            $csvArr = CSV_MTS_Manager::arrayFromCSV($filePath);
            return CSV_MTS_Manager::getTimeSerieFromArray($csvArr);
        }

        public static function getLegend_inCSV($filePath)
        {
            $csvArr = CSV_MTS_Manager::arrayFromCSV($filePath);
            return CSV_MTS_Manager::getDatesOrLegendFromField($csvArr);
        }
    }

    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------
    #-----------------------------------------------------------------------------------------------

    class CSV_TS_Manager
    {
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

        public static function getTimeSerieFromArray($csvArr)
        {
            $a = array();
            $ts = NULL;
            if(count($csvArr[0])==1)
            {
                foreach($csvArr as $arr)
                {
                    if(is_numeric($arr[0]))
                    {
                        $a[] = doubleval($arr[0]);
                    }
                }
                $ts = new TimeSerie($a);
            }
            else if(count($csvArr[0])==2)
            {
                foreach($csvArr as $arr)
                {
                    if(is_numeric($arr[1]))
                    {
                        $a[] = doubleval($arr[1]);
                    }
                }
                $ts = new TimeSerie($a);
            }
            return $ts;
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
                else if(doubleval($arr[0])==0)
                {
                    $legendArr[] = $arr[0];
                }
                else
                {
                    $legendArr[] = $arr[0];
                    //$doNotMatter = TRUE;
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

        /* *** CALLABLE *** */
        //get timeSerie in csv file
        //return timeSerie Object 
        public static function getTS_inCSV($filePath)
        {
            $csvArr = CSV_TS_Manager::arrayFromCSV($filePath);
            return CSV_TS_Manager::getTimeSerieFromArray($csvArr);
        }

        public static function getLegend_inCSV($filePath)
        {
            $csvArr = CSV_TS_Manager::arrayFromCSV($filePath);
            return CSV_TS_Manager::getDatesOrLegendFromField($csvArr);
        }
    }
    


?>