<?php
    define( 'WP_POST_REVISIONS', 6 );

    define("PATH_ROOT", dirname(__FILE__));
    

    require_once PATH_ROOT.'/../../models/dataPretreatmentService/timeSerie.php';

    class TSC_Manager
    {
        public static function getMaxArray($tsc){
            $maxArr = array();
            for($i=0;$i<$tsc->getTimeSerieCount();$i++){
                $maxArr[] = $tsc->getTS($i)->getMaxOfValues();
            }
            return $maxArr;
        }

        public static function getMinArray($tsc){
            $minArr = array();
            for($i=0;$i<$tsc->getTimeSerieCount();$i++){
                $minArr[] = $tsc->getTS($i)->getMinOfValues();
            }
            return $minArr;
        }

        public static function getNanArray($tsc){
            $nanArr = array();
            for($i=0;$i<$tsc->getTimeSerieCount();$i++){
                $nanArr[] = $tsc->getTS($i)->getNanValuesCount();
            }
            return $nanArr;
        }

        public static function getIndexArray($tsc){
            $indArr = array();
            for($i=0;$i<$tsc->getTimeSerieCount();$i++)
            {
                $indArr[] = $tsc->getIndexByIndex($i);
            }
            return $indArr;
        }

        public static function getIndexArray_column($rc){
            $indArr2 = array();
            for($i=0; $i<$rc->getTimeSerieCount();$i++){
                $indArr2[] = $rc->getIndexByIndex($i);
            }
            return $indArr2;
        }

        public static function getNanColumnIndex($tsc){
            $nanList = array();
            for($i=0;$i<$tsc->getTimeSerieCount();$i++){
                if($tsc->getTS($i)->isNanValuesInTS()==true)
                {
                    $nanList[] = $tsc->getIndexByIndex($i);
                }
            }
            return $nanList;
        }

        public static function getNotNanColumnIndex($tsc){
            $notNanList = array();
            for($i=0;$i<$tsc->getTimeSerieCount();$i++){
                if($tsc->getTS($i)->isNanValuesInTS()==false){
                    $notNanList[] = $tsc->getIndexByIndex($i);
                }
            }
            return $notNanList;
        }
    }

?>