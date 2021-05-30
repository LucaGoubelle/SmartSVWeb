<?php

    //author: Luca Goubelle

    require "controllers/dataPretreatmentService/csvTSDataManager.php";
    session_start();
    session_unset();
    session_start();

    if(isset($_POST['uploadButton']))
    {
        //
        var_dump($_FILES);
        //
        $fileName = $_FILES['notNormalisedFile']['name'];
        $fileTmpName = $_FILES['notNormalisedFile']['tmp_name'];
        $fileExtension = pathinfo($fileName, PATHINFO_EXTENSION);
        $allowedType = array('csv');
        if(!in_array($fileExtension, $allowedType))
        {
            ?><script>alert("ERR: wrong type of file, must be a .csv file !");</script><?php
            header('Location: ../vues/dataPretreatmentService/uploadTimeSerieFile.php');
        }
        else
        {
            //do csv
            $tsc = CSV_TS_Data_Manager::getTS_inCSV($fileTmpName);
            $rc = CSV_TS_Data_Manager::getRows_inCSV($fileTmpName);

            $uploadFolder = "data/dp_uploaded/";
            move_uploaded_file($fileTmpName, $uploadFolder.$fileName);

            $_SESSION['uploaded_fileName'] = $fileName;
            $_SESSION['uploaded_filePath'] = $fileTmpName;
            $_SESSION['tsc'] = $tsc;
            $_SESSION['rowCollection'] = $rc;
            header('Location: ../vues/dataPretreatmentService/columnsParameters.php');
        }
    }


?>