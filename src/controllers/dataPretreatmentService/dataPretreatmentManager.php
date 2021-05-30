<?php

	//author: Luca Goubelle

	class DataPretreatmentManager
	{

		public static function fillCSVColumns($fileName,$columns,$method)
		{
			
			//ob_start();
			passthru('python fillNan.py '.$fileName.' '.$columns.' '.$method.' 2>&1');
			$output = ob_get_clean();
    		ob_flush();
    		return $output;
		}

		public static function fillCSVColumns_linearRegress($fileName,$columns,$columnInvolved){
			//ob_start();
			passthru('python fillNanLinearRegress.py '.$fileName.' '.$columns.' '.$columnInvolved.' 2>&1');
			$output = ob_get_clean();
    		ob_flush();
    		return $output;
		}

		public static function normaliseDF($fileName, $indexCol)
		{
			//ob_start();
			passthru('python normalise01.py '.$fileName.' '.$indexCol.' 2>&1');
			$output = ob_get_clean();
    		ob_flush();
    		return $output;
		}

		public static function treatOutliers($fileName, $column, $method){
			//ob_start();
			passthru('python treatOutliers.py '.$fileName.' '.$column.' '.$method.' 2>&1');
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
        	$pathToUpload = getcwd()."data/dp_uploaded/";
        	$pathToFilled = getcwd()."data/filled/";
        	$pathToOutlierDone = getcwd()."data/outlierDone/";
        	$pathToSave = getcwd()."data/saved/";

        	$directories = array($pathToUpload, $pathToFilled, $pathToOutlierDone, $pathToSave);

        	foreach ($directories as $d) {
        		if(file_exists($d.$fileName)){
        			unlink($d.$fileName);
        		}
        	}
        }

	}

?>