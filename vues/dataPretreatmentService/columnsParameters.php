<?php

	require "../../src/models/dataPretreatmentService/timeSerie.php";
    require_once "../../src/controllers/dataPretreatmentService/TSCManager.php";

    session_start();

    $rc = isset($_SESSION["rowCollection"]) ? $_SESSION["rowCollection"] : NULL;
    $tsc = isset($_SESSION['tsc']) ? $_SESSION['tsc'] : NULL;
    $cmdOut = isset($_SESSION['cmdOut']) ? $_SESSION['cmdOut'] : NULL;
    $nrmF = isset($_SESSION['uploaded_fileName']) ? $_SESSION['uploaded_fileName'] : NULL;
    $nrmFP = isset($_SESSION['uploaded_filePath']) ? $_SESSION['uploaded_filePath'] : NULL;

    //var_dump($nrmF);
	echo "<br>";
	//var_dump($cmdOut);
    //var_dump($rc->getIndex());

    /*$maxArr = array();
    $minArr = array();
    $nanArr = array();
    $indArr = array();
    $indArr2 = array();*/

    $maxArr = TSC_Manager::getMaxArray($tsc);
    $minArr = TSC_Manager::getMinArray($tsc);
    $nanArr = TSC_Manager::getNanArray($tsc);
    $indArr = TSC_Manager::getIndexArray($tsc);
    $indArr2 = TSC_Manager::getIndexArray_column($rc);

    $nanList = TSC_Manager::getNanColumnIndex($tsc);
    $notNanList = TSC_Manager::getNotNanColumnIndex($tsc);

    /*
    for($i=0; $i<$rc->getTimeSerieCount();$i++){
        $indArr2[] = $rc->getIndexByIndex($i);
    }

    for($i=0;$i<$tsc->getTimeSerieCount();$i++)
    {
        $maxArr[] = $tsc->getTS($i)->getMaxOfValues();
        $minArr[] = $tsc->getTS($i)->getMinOfValues();
        $nanArr[] = $tsc->getTS($i)->getNanValuesCount();
    }

    for($i=0;$i<$tsc->getTimeSerieCount();$i++)
    {
        $indArr[] = $tsc->getIndexByIndex($i);
    }

    $nanList = array();
    for($i=0;$i<$tsc->getTimeSerieCount();$i++)
    {
        if($tsc->getTS($i)->isNanValuesInTS()==true)
        {
            $nanList[] = $tsc->getIndexByIndex($i);
        }
    }

    $notNanList = array();
    for($i=0;$i<$tsc->getTimeSerieCount();$i++)
    {
    	if($tsc->getTS($i)->isNanValuesInTS()==false)
    	{
    		$notNanList[] = $tsc->getIndexByIndex($i);
    	}
    }*/


    $nanListJS = json_encode($nanList);

    $totalNan = $tsc->getNanValuesCount();

?>
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>SmartSV Web</title>
        <link rel="stylesheet" type="text/css" href="../../css/bootstrap/css/bootstrap.min.css">
        <script src="../../js/pretreatParam.js"></script>
    </head>
    <body>
        <form action="../../src/quit.php" method="post"><input class="btn btn-danger" name="quitButton" value="Quit" type="submit"></form>
        <form action="../../src/downloadTS.php" method="post">
            <h1>General information of TS</h1>
            <p>Number of column : <?php echo $tsc->getTimeSerieCount(); ?></p>
            <p>Number of lines : <?php echo $tsc->getNumberOfTSValues(); ?></p>
            <p><strong>index column of TS</strong></p>
            <select class="form-select" name="columnIndex">
            <?php
                for($i=0;$i<count($indArr2);$i++){
                    echo "<option value=\"".$indArr2[$i]."\">".$indArr2[$i]."</option>";
                }
            ?>
            </select>
            <p>Number of NAN values : <?php echo $tsc->getNanValuesCount(); ?></p>
            <p>Numeric columns : <?php echo implode(",", $tsc->getIndex()); ?></p>
            <p>Detailed columns : </p>
            
            <table style="text-align: center;" id="tableColumnsContent">
                <!--<thead>-->
                    <tr>
                        <th>Column</th>
                        <th>Max value</th>
                        <th>Min value</th>
                        <th>Number of NAN</th>
                    </tr>
            <!-- </thead>-->
                <!--<tbody>-->
                    <?php 
                        for($i=0;$i<count($nanArr);$i++)
                        {
                            ?><tr> <td><?php echo $indArr[$i]; ?></td> <td><?php echo $maxArr[$i]; ?></td> <td><?php echo $minArr[$i]; ?></td> <td><?php echo $nanArr[$i]; ?><td/></tr><?php
                        }
                    ?>
                <!--</tbody>-->
            </table><br>
            <a href="treatNanValues.php"><input style="color: white;" class="btn btn-info" value="Treat NAN values"></a><br><br><br>
            <a href="treatOutliers.php"><input style="color: white;" class="btn btn-info" value="Treat Outliers values"></a><br><br><br>
            
            <br><br>
            <input type="submit" name="downloadTS" class="btn btn-success" value="save df">
            <input type="submit" name="dfNormalise01" class="btn btn-success" value="save df01">
        </form>
            

        <?php 
            $_SESSION['lstColNan'] = $nanList;
            $_SESSION['lstColNotNan'] = $notNanList;
            $_SESSION['totalNan'] = $totalNan;
        ?>


    <script>

    </script>
    </body>
</html>