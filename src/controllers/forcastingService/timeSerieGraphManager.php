<?php

    //author: Luca Goubelle

    require_once "csvTSManager.php";
    require_once "/../../models/forcastingService/timeSerieChart.php";

    class TimeSerieGraphManager
    {
        public static function generateGraphData($lg, $ts)
        {
            return new SingleTimeSerieChart($lg, $ts);
        }

        public static function generateMultiGraphData($lg, $ts)
        {
            return new MultiTimeSerieChart($lg, $ts);
        }
    }

?>