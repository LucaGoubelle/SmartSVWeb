<?php

    //author: Luca Goubelle

	
    require "controllers/forcastingService/csvTSManager.php";
    session_start();
    session_unset();
    session_start();

    if(isset($_POST['uploadButton']))
    {
        //
        //var_dump($_FILES);
        //

        $fileName = $_FILES['timeSerieFile']['name'];
        $fileTmpName = $_FILES['timeSerieFile']['tmp_name'];
        $fileExtension = pathinfo($fileName, PATHINFO_EXTENSION);
        $allowedType = array('csv');
        if(!in_array($fileExtension, $allowedType))
        {
            header('Location: ../vues/forcastingService/uploadTimeSerie.php');
            ?><script>alert("ERR: wrong type of file, must be a .csv file !");</script><?php
        }
        else
        {
            //do csv
            $tsc = CSV_MTS_Manager::getMTS_inCSV($fileTmpName);
            $id = CSV_MTS_Manager::getDatesOrLegendFromField($fileTmpName);

            $uploadFolder = "data/fs_uploaded/";
            move_uploaded_file($fileTmpName, $uploadFolder.$fileName);

            $_SESSION['uploaded_fileName'] = $fileName;
            $_SESSION['uploaded_filePath'] = $fileTmpName;
            $_SESSION['tsc'] = $tsc;
            $_SESSION['id'] = $id;
            header('Location: ../vues/forcastingService/showTimeSerie.php');
        }
    }


?>