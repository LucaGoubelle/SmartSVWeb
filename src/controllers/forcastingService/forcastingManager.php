<?php

    //author: Luca Goubelle

    class ForcastingManager
    {
    	public static function classicForcasting($nameFile){
    		passthru("python classicForcasting.py ".strval($nameFile)." 2>&1");
    		$output = ob_get_clean();
    		ob_flush();
    		return $output;
    	}

        public static function forcasting($nameFile, $methodsStr, $nbPastValues, $nbValuesToPredict)
        {
            passthru("python forcastingOneSerie.py ".strval($nameFile)." ".strval($methodsStr)." ".strval($nbPastValues)." ".strval($nbValuesToPredict)." 2>&1");
            $output = ob_get_clean();
            ob_flush();
    		return $output;
        }

        public static function multiForcasting($nameFile, $algos, $nbPrev, $longPred, $percTrain, $x, $y){
            passthru("python forcastingMultiSerie.py ".strval($nameFile)." ".strval($algos)." ".strval($nbPrev)." ".strval($longPred)." ".strval($percTrain)." ".strval($x)." ".strval($y)." 2>&1");
            $output = ob_get_clean();
            ob_flush();
            return $output;
        }

        public static function downloadFile($file) { // $file = include path
            if(file_exists($file)) {
                header('Content-Description: File Transfer');
                header('Content-Type: application/octet-stream');
                header('Content-Disposition: attachment; filename='.basename($file));
                header('Content-Transfer-Encoding: binary');
                header('Expires: 0');
                header('Cache-Control: must-revalidate, post-check=0, pre-check=0');
                header('Pragma: public');
                header('Content-Length: ' . filesize($file));
                ob_clean();
                flush();
                readfile($file);
                exit;
            }
        }

        public static function clearFolders($fileName)
        {
            $pathToUpload = "data/fs_uploaded/";
            $pathToForcasted = "data/forcasted/";

            $directories = array($pathToUpload, $pathToForcasted);

            foreach ($directories as $d) {
                if(file_exists($d.$fileName)){
                    unlink($d.$fileName);
                }
            }
        }
    }

?>