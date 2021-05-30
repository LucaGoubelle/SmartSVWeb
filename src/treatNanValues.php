<?php
    ob_start();

    require_once "controllers/dataPretreatmentService/dataPretreatmentManager.php";
    require_once "controllers/dataPretreatmentService/csvTSDataManager.php";
    require_once "controllers/dataPretreatmentService/TSCManager.php";
    session_start();

    //$fileName = $_SESSION['fileName'];
    $fileName = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;

    if(isset($_POST["treat"])){
        $cmdOut = "";
        switch($_POST["columnMethodName"])
        {
            case "OCB": $cmdOut = DataPretreatmentManager::fillCSVColumns($fileName, $_POST["columnName"], $_POST["columnMethodName"]); break;
            case "LOCF": $cmdOut = DataPretreatmentManager::fillCSVColumns($fileName, $_POST["columnName"], $_POST["columnMethodName"]); break;
            case "mean": $cmdOut = DataPretreatmentManager::fillCSVColumns($fileName, $_POST["columnName"], $_POST["columnMethodName"]); break;
            default: break;
        }
        $_SESSION['cmdOut'] = $cmdOut;
        $_SESSION['tsc'] = CSV_TSFill_Data_Manager::getTS_inCSV("data/filled/".$fileName);
        $_SESSION['lstColNan'] = TSC_Manager::getNanColumnIndex($_SESSION['tsc']);
        $_SESSION['lstColNotNan'] = TSC_Manager::getNotNanColumnIndex($_SESSION['tsc']);
        header('Location: ../vues/dataPretreatmentService/treatNanValues.php');
    }

    ob_end_flush();
?>