<?php

    //author: Luca Goubelle

    require "csvTSManager.php";
    require "csvTSFManager.php";
    require_once "/../../models/forcastingService/timeSerieForcastedChart.php";

    class ForcastedSerieGraphManager
    {
        public static function generateGraphSimpleForcastedData($file, $fileResult)
        {
            $legend = CSV_TS_Manager::getLegend_inCSV($file);
            $originalTS = CSV_TS_Manager::getTS_inCSV($file);
            $fTS = CSV_TSF_Manager::getTSF_inCSV($fileResult);

            return new SingleTimeSerieForcastedChart($legend,$originalTS,$fTS);
        }

        public static function generateGraphMultiForcastedData($file, $fileResult)
        {
            $legend = CSV_MTS_Manager::getLegend_inCSV($file);
            $originalsTSs = CSV_MTS_Manager::getMTS_inCSV($file);
            $fTSs = CSV_MTSF_Manager::getMTSF_inCSV($fileResult);

            return new MultiTimeSerieForcastedChart($legend, $originalsTSs, $fTSs);
        }
    }

?>