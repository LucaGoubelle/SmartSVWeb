<?php
    ob_start();

	require_once "controllers/dataPretreatmentService/dataPretreatmentManager.php";
    require_once "controllers/dataPretreatmentService/csvTSDataManager.php";
    session_start();

    //$fileName = $_SESSION['fileName'];
    $fileName = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;

    if(isset($_POST["treatOutliers"])){
    	$cmdOut = "";
        switch($_POST["methodName"])
        {
            case "EllipticEnvelope": $cmdOut = DataPretreatmentManager::treatOutliers($fileName, $_POST["columnName"], $_POST["methodName"]); break;
            case "IsolationForest": $cmdOut = DataPretreatmentManager::treatOutliers($fileName, $_POST["columnName"], $_POST["methodName"]); break;
            case "LocalOutlierFactor": $cmdOut = DataPretreatmentManager::treatOutliers($fileName, $_POST["columnName"], $_POST["methodName"]); break;
            case "OneClassSVM": $cmdOut = DataPretreatmentManager::treatOutliers($fileName, $_POST["columnName"], $_POST["methodName"]); break;
            default: break;
        }
        $_SESSION['cmdOut'] = $cmdOut;
        $_SESSION['tsc'] = CSV_TSFill_Data_Manager::getTS_inCSV("data/outlierDone/".$fileName);
        copy("data/outlierDone/".$fileName, "data/saved/".$fileName);
        ?><script>alert("Outliers done !");</script><?php
        var_dump($cmdOut);
        header('Location: ../vues/dataPretreatmentService/treatOutliers.php');
    }

    ob_end_flush();
?>