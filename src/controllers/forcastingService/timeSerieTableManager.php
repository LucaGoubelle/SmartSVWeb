<?php

    class TimeSerieTableManager
    {
        public static function showTimeSerieTable($ts){
            $content = "<table>";
            $arr = $ts->getValues();
            $content .= "<tr><td style='color: lime;'>TS_</td>";
            for($i=0;$i<count($arr);$i++){
                $content .= "<td><p>[ <strong style='color: cyan;'>".$arr[$i]."</strong> ] , </p></td>";
            }
            $content .= "</tr></table>";
            echo $content;
        }

        public static function showTimeSerieIndexTable($arr){
            $content = "<table>";
            $content .= "<tr><td style='color: lime;'>TS_IND</td>";
            for($i=0;$i<count($arr);$i++){
                $content .= "<td><p>[ <strong style='color: cyan;']".$arr[$i]."</strong> ] , </p></td>";
            }
            $content .= "</tr></table>";
            echo $content;
        }

        public static function showTimeSerieForcastedTable($ts){
            $content = "<table>";
            $arr = $ts->getValues();
            $content .= "<tr><td style='color: lime;'>TS_FCST</td>";
            for($i=0;$i<count($arr);$i++){
                $content .= "<td><p>[ <strong style='color: red;'>".$arr[$i]."</strong> ] , </p></td>";
            }
            $content .= "</tr></table>";
            echo $content;
        }

        public static function showTimeSerieCollectionTable($tsc){
            $content = "<table>";
            $arr = array();
            $content .= "<tr>";
            for($i=0;$i<$tsc->getTimeSerieCount();$i++){ $arr[] = $tsc->getTS($i); $content .= "<th style='color: lime;'>TS_". $i+1 ."</th>"; }
            $content .= "</tr>";

            for($i=0;$i<count($arr[0]);$i++){
                $content .= "<tr>";
                for($j=0;$j<count($arr);$j++){
                    $content .= "<td style='color: cyan;'>".$arr[$j][$i]."</td>";
                }
                $content .= "</tr>";
            }
            $content .= "</table>";
            echo $content;
        }
    }

?>