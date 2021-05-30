<?php
    ob_start();

    //author: Luca Goubelle

    require "controllers/forcastingService/forcastingManager.php";
    require "controllers/forcastingService/csvTSFManager.php";
    session_start();

    $upload_fileName = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;
    //$normalised_filePath = isset($_SESSION['normalised_filePath']) ? $_SESSION['normalised_filePath'] : NULL;

    if(isset($_POST['classicPredictButton'])){
        $cmdOut = ForcastingManager::classicForcasting($upload_fileName);
        $_SESSION['cmdOut'] = $cmdOut;
        $_SESSION['fcst'] = CSV_TSF_Manager::getTSF_inCSV("data/forcasted/".$upload_fileName);
        header('Location: ../vues/forcastingService/showTimeSerie.php');
    }
    else if(isset($_POST["submit"]) and isset($_POST['previousValueField']) and isset($_POST['predictValsField']) and isset($_POST['percentTraining']) and isset($_POST['yCol']))
    {
        if(!empty($_POST["method"]) and !empty($_POST["xCols"]))
        {
            $listMethod = "";
            foreach($_POST["method"] as $method)
            {
                $listMethod .= $method;
                $listMethod .= ",";
            }
            $listMethod = substr($listMethod, 0, -1);

            $listX = "";
            foreach($_POST["xCols"] as $col){
                $listX .= $col;
                $listX .= ",";
            }
            $listX = substr($listX, 0, -1);
            
            $cmdOut = ForcastingManager::multiForcasting($upload_fileName,  $listMethod, strval($_POST['previousValueField']), strval($_POST['predictValsField']), strval($_POST['percentTraining']), $listX, strval($_POST['yCol']));
            $_SESSION['cmdOut'] = $cmdOut;
            //echo $cmdOut;
            $_SESSION['fcst'] = CSV_TSF_Manager::getTSF_inCSV("data/forcasted/".$upload_fileName);
            header('Location: ../vues/forcastingService/showTimeSerie.php');
        }
    }

    ob_end_flush();
?>