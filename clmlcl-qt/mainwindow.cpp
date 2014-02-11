#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QTableWidgetItem>
#include <iostream>
#include "matrixdialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    cls(*new classifier)
{
    ui->setupUi(this);
    srand(time(0));
}

void MainWindow::showmatrix(const matrix& x, const char* title)
{
    MatrixDialog* md = new MatrixDialog(this, x);
    if (title) md->setWindowTitle(title);
    md->show();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()
{
    cls.run_test(
                ui->indimEdit->text().toInt(),
                ui->outdimEdit->text().toInt(),
                ui->lambdaEdit->text().toFloat(),
                ui->kernelCombo->currentText(),
                ui->ntestEdit->text().toInt(),
                ui->ntrainEdit->text().toInt(),
                ui->nbatchEdit->text().toInt(),
                ui->nitersEdit->text().toInt()
    );
}

void MainWindow::on_pushButton_2_clicked()
{
    showmatrix(cls.trainset.first);
}

void MainWindow::on_pushButton_3_clicked()
{
    showmatrix(cls.trainset.second);
}

void MainWindow::on_pushButton_4_clicked()
{
    showmatrix(cls.train_hyperplane);
}

void MainWindow::on_pushButton_5_clicked()
{
    showmatrix(cls.alpha);
}

void MainWindow::on_pushButton_6_clicked()
{
    cls.test(ui->kernelCombo->currentText());
}
