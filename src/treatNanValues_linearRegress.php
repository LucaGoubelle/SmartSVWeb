<?php
    ob_start();

    require_once "controllers/dataPretreatmentService/dataPretreatmentManager.php";
    require_once "controllers/dataPretreatmentService/csvTSDataManager.php";
    require_once "controllers/dataPretreatmentService/TSCManager.php";
    session_start();

    $fileName = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;

    if(isset($_POST["treatNanLR"])){
        if(!empty($_POST["columns"])){
            $listCols = "";
            foreach($_POST["columns"] as $method)
            {
                $listCols .= $method;
                $listCols .= ",";
            }
            $listCols = substr($listCols, 0, -1);
            $cmdOut = DataPretreatmentManager::fillCSVColumns_linearRegress($fileName, $listCols, $_POST["columnName"]);
            $_SESSION['cmdOut'] = $cmdOut;
            $_SESSION['tsc'] = CSV_TSFill_Data_Manager::getTS_inCSV("data/filled/".$fileName);
            $_SESSION['lstColNan'] = TSC_Manager::getNanColumnIndex($_SESSION['tsc']);
            $_SESSION['lstColNotNan'] = TSC_Manager::getNotNanColumnIndex($_SESSION['tsc']);
            header('Location: ../vues/dataPretreatmentService/treatNanValues.php');
        }
    }

    ob_end_flush();
?>